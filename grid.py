import numpy as np
from shapely.geometry import Polygon, box

def grid_placement(points,scale,W,overlap,margin,cell_width,cell_height,offset_x,offset_y,gap=10):
    result={'status':'보류: 스케일 미확정','area_um2':None,'layout':None,'parts':{}}
    if scale<=0:return result
    flake=Polygon(np.asarray(points)*scale)
    safe=flake.buffer(-margin)
    result.update(area_um2=round(flake.area,3),status='현재 격자 위치에서 양쪽 전극 접촉 미확보')
    if overlap>min(cell_width,cell_height):return result
    xmin,ymin,xmax,ymax=flake.bounds
    px=cell_width+gap;py=cell_height+gap
    for i in range(int(np.floor((xmin-offset_x)/px))-1,int(np.ceil((xmax-offset_x)/px))+1):
        for j in range(int(np.floor((ymin-offset_y)/py))-1,int(np.ceil((ymax-offset_y)/py))+1):
            x=offset_x+i*px;y=offset_y+j*py
            for axis in ['수평','수직']:
                low=max(y,ymin) if axis=='수평' else max(x,xmin)
                high=min(y+cell_height,ymax) if axis=='수평' else min(x+cell_width,xmax)
                for center in np.linspace(low+W/2,high-W/2,17) if high-low>=W else []:
                    if axis=='수평':
                        edge=x+cell_width
                        parts={'source':box(edge-overlap,center-W/2,edge,center+W/2),'channel':box(edge,center-W/2,edge+gap,center+W/2),'drain':box(edge+gap,center-W/2,edge+gap+overlap,center+W/2)}
                        footprint=box(edge-overlap,center-W/2,edge+gap+overlap,center+W/2)
                    else:
                        edge=y+cell_height
                        parts={'source':box(center-W/2,edge-overlap,center+W/2,edge),'channel':box(center-W/2,edge,center+W/2,edge+gap),'drain':box(center-W/2,edge+gap,center+W/2,edge+gap+overlap)}
                        footprint=box(center-W/2,edge-overlap,center+W/2,edge+gap+overlap)
                    if safe.covers(footprint):
                        result.update(status='격자 틈을 가로질러 양쪽 전극 접촉 가능',layout=(np.asarray(footprint.exterior.coords)/scale).tolist(),parts={k:(np.asarray(v.exterior.coords)/scale).tolist() for k,v in parts.items()},angle_deg=0 if axis=='수평' else 90)
                        return result
    return result

def draw_grid(image,scale,cw,ch,ox,oy,gap=10):
    from PIL import ImageDraw
    out=image.copy();draw=ImageDraw.Draw(out)
    if scale<=0:return out
    for i in range(int(np.floor(-ox/(cw+gap)))-1,int(np.ceil((image.width*scale-ox)/(cw+gap)))+1):
        for j in range(int(np.floor(-oy/(ch+gap)))-1,int(np.ceil((image.height*scale-oy)/(ch+gap)))+1):
            x=ox+i*(cw+gap);y=oy+j*(ch+gap)
            draw.rectangle((x/scale,y/scale,(x+cw)/scale,(y+ch)/scale),outline='#a0a0ff',width=1)
    return out

def auto_align_grid(candidates, scale, W, overlap, margin, cell_width, cell_height):
    """Search shared grid translations around uniform candidates; never rotate cells."""
    from assessment import assess
    if scale <= 0:
        return {'ok': False, 'reason': '스케일을 먼저 확인하세요.'}
    order = sorted(enumerate(candidates), key=lambda item: (
        item[1].get('residue_status', '판단 보류') == '판단 보류',
        {'얇은 영역 추정': 0, '판단 불가': 1, '두꺼운 영역 추정': 2}.get(item[1].get('thin_hint'), 1),
        not item[1].get('reviewed', False), item[0]))
    for index, candidate in order:
        if candidate.get('uniformity') != '균일 추정':
            continue
        if candidate.get('internal_gap') and not candidate.get('continuity_confirmed'):
            continue
        flake = Polygon(np.asarray(candidate['points']) * scale)
        safe = flake.buffer(-margin)
        if safe.is_empty:
            continue
        xmin, ymin, xmax, ymax = safe.bounds
        centers = [(safe.representative_point().x, safe.representative_point().y)]
        centers += [(x,y) for x in np.linspace(xmin,xmax,11) for y in np.linspace(ymin,ymax,11)]
        for x,y in centers:
            for horizontal in [True, False]:
                half_length = 5 + overlap
                rect = box(x-half_length,y-W/2,x+half_length,y+W/2) if horizontal else box(x-W/2,y-half_length,x+W/2,y+half_length)
                if not safe.covers(rect):
                    continue
                ox = (x-5-cell_width) % (cell_width+10) if horizontal else (x-cell_width/2) % (cell_width+10)
                oy = (y-cell_height/2) % (cell_height+10) if horizontal else (y-5-cell_height) % (cell_height+10)
                geometry = grid_placement(candidate['points'],scale,W,overlap,margin,cell_width,cell_height,ox,oy)
                if not geometry['layout']:
                    continue
                trial = dict(candidate, geometry=geometry, grid_confirmed=True)
                evaluation = assess(trial)
                # Unknown residue remains explicitly provisional, not an all-clear.
                provisional = candidate.get('residue_status','판단 보류') == '판단 보류'
                permitted = evaluation['eligible'] or (provisional and evaluation['reasons'] == ['잔류 Au/오염 의심 여부 확인 필요'])
                if permitted:
                    return {'ok':True,'candidate_index':index,'offset_x_um':float(ox),'offset_y_um':float(oy),'provisional':provisional,'reason':'균일성·잔류물 상태·얇기 순으로 검사하여 양쪽 전극 접촉을 확보한 첫 후보','geometry':geometry}
    return {'ok':False,'reason':'탐색한 위치에서 조건을 만족하는 후보를 찾지 못했습니다. 균일성·연속성·크기·잔류물 판단을 확인하세요.'}
