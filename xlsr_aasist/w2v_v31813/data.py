"""Existing augmentation unchanged; original PCL references travel separately."""
from w2v_v31810.data import Waves as BaseWaves
from w2v_v3189.data import prepared
from .phonemes import nonsilent

class Waves(BaseWaves):
    def __init__(self,rows,cfg,split='train'):
        super().__init__(rows,cfg,split);self.by_id={r['id']:r for r in rows}

    def __getitem__(self,ticket):
        out=super().__getitem__(ticket)
        if isinstance(ticket,int):return out
        row=out[0]
        if 'xeus_pair' in ticket:
            row['xeus_pair']=ticket['xeus_pair'];row['xeus_similarity']=ticket['xeus_similarity']
        if 'official_pcl' in ticket:
            p=ticket['official_pcl'];a,b=self.by_id[p['offline']],self.by_id[p['online']]
            if a['canonical_source']!=b['canonical_source'] or a['label']!=b['label']:raise ValueError('Invalid PCL reference')
            wa,wb=self.original(a),self.original(b);matches=nonsilent(p['matches'],wa,wb)
            if matches:
                row['original_pcl']=dict(refs=[prepared(a,wa),prepared(b,wb)],matches=matches)
            else:row['original_pcl_empty']=True
        return out
