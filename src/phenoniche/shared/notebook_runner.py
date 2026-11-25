"""Model-independent background notebook execution with live persisted streams."""
from pathlib import Path
import fcntl
import os
import time
import traceback
from .io import output_path, write_json


def execute_notebook(notebook, output, *, kernel_name='cccphe', workdir=None,
                     start_cell=0, setup_source=None):
    import nbformat
    from nbclient import NotebookClient
    notebook = Path(notebook).resolve()
    output = output_path(output)
    with (output / 'worker.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / 'worker.pid').write_text(str(os.getpid()) + '\n')
        book = nbformat.read(notebook, as_version=4)
        write_json(output / 'active_run.json', dict(notebook=str(notebook), pid=os.getpid(),
                   backend=book.metadata['model_backend'], physical_gpu=book.metadata['physical_gpu'],
                   start_cell=start_cell + 1,
                   scope='Clustering/display refresh from saved fit; no training' if start_cell else
                   'All notebook cells: spatial epoch selection, full refit, clustering and result figures'))

        class StreamingClient(NotebookClient):
            def process_message(self, message, cell, cell_index):
                stream = message.get('msg_type', message.get('header', {}).get('msg_type')) == 'stream'
                if stream:
                    print(message['content']['text'], end='', flush=True)
                result = super().process_message(message, cell, cell_index)
                if stream and (time.monotonic() - getattr(self, 'last_persist', 0) > 15
                               or any(token in message['content']['text'] for token in
                                      ('GNN epoch ',))):
                    persist()
                    self.last_persist = time.monotonic()
                return result

        def persist():
            live = nbformat.read(notebook, as_version=4)
            executed = {cell.id: cell for cell in book.cells}
            for cell in live.cells:
                saved = executed.get(cell.id)
                if saved is not None and cell.cell_type == 'code' and cell.source == saved.source:
                    cell.outputs, cell.execution_count = saved.outputs, saved.execution_count
            temporary = notebook.with_suffix('.running.tmp')
            nbformat.write(live, temporary)
            temporary.replace(notebook)

        client = StreamingClient(book, timeout=None, kernel_name=kernel_name,
                                 resources={'metadata': {'path': str(workdir or notebook.parent)}})
        try:
            with client.setup_kernel():
                if setup_source:
                    first = book.cells[0]
                    try:
                        client.execute_cell(nbformat.v4.new_code_cell(setup_source), 0)
                    finally:
                        book.cells[0] = first
                for index, cell in enumerate(book.cells):
                    if index < start_cell or cell.cell_type != 'code':
                        continue
                    write_json(output / 'status.json', dict(status='running', cell=index + 1,
                               heading=cell.source.splitlines()[0], pid=os.getpid(), notebook=str(notebook)))
                    print(f'Cell {index + 1}: {cell.source.splitlines()[0]}', flush=True)
                    try:
                        client.execute_cell(cell, index, execution_count=index + 1)
                    finally:
                        persist()
            write_json(output / 'status.json', dict(status='complete', pid=os.getpid(), notebook=str(notebook)))
            print('Notebook complete:', notebook, flush=True)
        except BaseException as error:
            write_json(output / 'status.json', dict(status='failed', pid=os.getpid(), error=str(error)))
            traceback.print_exc()
            raise
