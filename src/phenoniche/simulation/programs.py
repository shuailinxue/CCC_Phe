"""Explicit synthetic MI identities and smooth spatial activations, before CCC."""
import numpy as np


def program_specs(config):
    specs=[{'MI':1,'name':'Edge A','region':2,'sender':config.sender_type,'receiver':config.receiver_type},
           {'MI':2,'name':'Edge B','region':3,'sender':config.sender_type,'receiver':config.receiver_type}]
    if config.program_design=='four_programs':
        specs += [{'MI':3,'name':'Tumor core','region':0,'sender':'Tumor','receiver':'Tumor'},
                  {'MI':4,'name':'Normal exterior','region':1,'sender':'Normal epithelial','receiver':'Normal epithelial'}]
    return specs


def spatial_activity(adata,config):
    """Cell-level design; downstream physical pairs yield mixed neighborhood activity."""
    radius=adata.obs.radius_um.to_numpy();x=adata.obs.local_x.to_numpy()
    outer=config.core_radius_um+config.edge_width_um
    edge=(radius>=config.core_radius_um)&(radius<outer)
    edge_profile=np.exp(-.5*((radius-(config.core_radius_um+config.edge_width_um/2))/(config.edge_width_um/3))**2)*edge
    result=np.zeros((len(adata),config.mi_count),np.float32)
    result[:,0]=edge_profile*(x<0);result[:,1]=edge_profile*(x>=0)
    if config.mi_count==4:
        transition=config.edge_width_um/2
        result[:,2]=np.clip((config.core_radius_um-radius)/transition,0,1)
        result[:,3]=np.clip((radius-outer)/transition,0,1)
    return result
