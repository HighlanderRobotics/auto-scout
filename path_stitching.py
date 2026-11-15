"""Reconnect uniquely compatible tracker fragments and mark short estimated gaps."""
import math


def continuation_error(first, second, max_gap=2.0):
    """Predict a fragment's entry from a short history, rather than one noisy step."""
    a,b=first['points'][-1],second['points'][0]
    gap=b['t']-a['t']
    if not 0<gap<=max_gap:
        return None
    history=[p for p in first['points'] if 0<a['t']-p['t']<=.5]
    vx=vy=0
    if history:
        prior=history[0];dt=a['t']-prior['t']
        vx=(a['x']-prior['x'])/dt;vy=(a['y']-prior['y'])/dt
        speed=math.hypot(vx,vy)
        if speed>35:
            vx*=35/speed;vy*=35/speed
    error=math.hypot(b['x']-a['x']-vx*gap,b['y']-a['y']-vy*gap)
    if error>min(15,4+8*gap) or math.hypot(b['x']-a['x'],b['y']-a['y'])>35*gap+3:
        return None
    return error


def stitch_fragments(tracks, colors, identities, blocked=()):
    """Mutual, unambiguous links; reject conflicts across the whole joined chain."""
    by_id={t['id']:t for t in tracks if t['points']}
    chains={i:[i] for i in by_id}
    owner={i:i for i in by_id}
    candidates=[]
    for i,first in by_id.items():
        if i in blocked or not colors.get(i):continue
        for j,second in by_id.items():
            if j in blocked or colors.get(i)!=colors.get(j):continue
            if identities.get(i) and identities.get(j) and identities[i]!=identities[j]:continue
            error=continuation_error(first,second)
            if error is not None:candidates.append((error,i,j))
    # Prefer an uninterrupted chain over a skip that bypasses an intervening
    # compatible fragment. Competing fragments at the same time remain ambiguous.
    edges={(i,j) for _,i,j in candidates}
    candidates=[(error,i,j) for error,i,j in candidates if not any(
        (i,k) in edges and (k,j) in edges and
        by_id[k]['points'][-1]['t']<by_id[j]['points'][0]['t'] for k in by_id)]
    outgoing={};incoming={}
    for error,i,j in sorted(candidates):
        outgoing.setdefault(i,[]).append((error,j))
        incoming.setdefault(j,[]).append((error,i))
    audit=[]
    for error,i,j in sorted(candidates):
        if outgoing[i][0][1]!=j or incoming[j][0][1]!=i:continue
        if len(outgoing[i])>1 and outgoing[i][1][0]-error<2:continue
        if len(incoming[j])>1 and incoming[j][1][0]-error<2:continue
        left,right=owner[i],owner[j]
        if left==right:continue
        a,b=chains[left],chains[right]
        if a[-1]!=i or b[0]!=j:continue
        names={identities[k] for k in a+b if k in identities}
        if len(names)>1:continue
        # A person/duplicate track can otherwise bridge two simultaneous robots.
        if any(max(by_id[x]['points'][0]['t'],by_id[y]['points'][0]['t'])<=
               min(by_id[x]['points'][-1]['t'],by_id[y]['points'][-1]['t']) for x in a for y in b):continue
        chains[left]=a+b;del chains[right]
        for k in b:owner[k]=left
        audit.append({'from_id':i,'to_id':j,'reason':'unique mutual motion continuation with matching bumper color','prediction_error':round(error,3)})
    return list(chains.values()),audit


def fill_short_gaps(points, fps, max_gap=2):
    """Only interpolate bounded gaps; never invent positions before/after visibility."""
    output=[]
    for point in points:
        if output:
            previous=output[-1];gap=point['t']-previous['t']
            fragment_change=point.get('source_id')!=previous.get('source_id')
            if (fragment_change or gap>.6) and 1.5/fps<gap<=max_gap and math.hypot(point['x']-previous['x'],point['y']-previous['y'])<=25*gap:
                steps=math.ceil(gap*fps)
                for step in range(1,steps):
                    fraction=step/steps
                    output.append({'t':round(previous['t']+gap*fraction,4),
                        'x':previous['x']+(point['x']-previous['x'])*fraction,
                        'y':previous['y']+(point['y']-previous['y'])*fraction,
                        'estimated':True,'source_ids':[previous.get('source_id'),point.get('source_id')]})
        output.append(dict(point))
    return output
