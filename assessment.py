"""Research selection policy; OM appearance is not a confirmed thickness/material assay."""
from shapely.geometry import Polygon

POLICY_VERSION='color-uniformity-grid-v2'
UNIFORMITY=['판단 보류','균일 추정','층수 경계 의심','여러 층수 확인']
RESIDUE=['판단 보류','의심 소견 없음','잔류 Au/오염 의심']

def assess(candidate):
    uniform=candidate.get('uniformity','판단 보류')
    residue=candidate.get('residue_status','판단 보류')
    regions=candidate.get('residue_regions',[])
    g=candidate.get('geometry',{})
    overlaps=[]
    invalid=False
    for region in regions:
        try:
            p=Polygon(region['points'])
            if not p.is_valid or p.area<=0:raise ValueError('invalid region')
            for name,points in g.get('parts',{}).items():
                if p.intersects(Polygon(points)) and name not in overlaps:overlaps.append(name)
        except (ValueError,KeyError,TypeError):invalid=True
    reasons=[]
    if candidate.get('grid_confirmed') is False:reasons.append('격자 마스크 치수·정렬 위치 미확정: 가정 배치')
    excluded=uniform in ['층수 경계 의심','여러 층수 확인']
    if excluded:reasons.append('동일 결정 내부의 층수 경계/혼합 층수 때문에 기본 추천 제외')
    elif uniform!='균일 추정':reasons.append('결정 전체의 층수 균일성 확인 필요')
    if not g.get('layout'):reasons.append('스케일·연속 영역·10 μm 배치 확인 필요')
    if candidate.get('internal_gap') and not candidate.get('continuity_confirmed'):
        reasons.append('내부 단절 감지: 연구자의 연속성 확인 필요')
    if regions and residue!='잔류 Au/오염 의심':reasons.append('표면 상태와 의심 영역 표시가 불일치: 재검토 필요')
    if invalid:reasons.append('잔류물 의심 영역 좌표 오류: 재검토 필요')
    if residue not in RESIDUE or residue=='판단 보류':reasons.append('잔류 Au/오염 의심 여부 확인 필요')
    if overlaps:
        labels={'source':'소스 접촉','channel':'채널','drain':'드레인 접촉'}
        reasons.append('잔류물 의심 영역이 '+', '.join(labels[x] for x in overlaps)+'과 겹침')
    elif residue=='잔류 Au/오염 의심' and not regions:
        reasons.append('잔류물 의심 위치 미지정: 채널/접촉 중첩 판단 보류')
    status='기본 추천 제외' if excluded else ('확인 대기' if reasons else '조건부 추천')
    return {'policy_version':POLICY_VERSION,'scope':'동일 결정 전체','status':status,'eligible':status=='조건부 추천','reasons':reasons or ['균일 추정·기하 배치 충족·채널/접촉 영역에 표시된 잔류물 중첩 없음'],'residue_overlap':overlaps,'residue_outside_layout':bool(regions) and not overlaps and not invalid,'certainty':'OM/사용자 검토 기반. 동일 층수 및 Au 성분 확정 아님'}

def rank_candidates(candidates):
    eligible=[c for c in candidates if assess(c)['eligible']]
    return sorted(eligible,key=lambda c:({'얇은 영역 추정':0,'판단 불가':1,'두꺼운 영역 추정':2}.get(c.get('thin_hint'),1),not c.get('reviewed',False)))
