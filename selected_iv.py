import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Polygon
from core import cox

def candidate_figure(image,candidates):
    figure=go.Figure()
    figure.add_layout_image(dict(source=image,x=0,y=0,sizex=image.width,sizey=image.height,xref='x',yref='y',sizing='stretch',layer='below',xanchor='left',yanchor='top'))
    for index,candidate in enumerate(candidates):
        points=np.asarray(candidate['points']);center=Polygon(points).representative_point()
        figure.add_trace(go.Scatter(x=list(points[:,0])+[points[0,0]],y=list(points[:,1])+[points[0,1]],mode='lines',line=dict(color='#17b8a5',width=2),hoverinfo='skip',showlegend=False))
        figure.add_trace(go.Scatter(x=[center.x],y=[center.y],mode='markers+text',text=[str(index+1)],textposition='top center',customdata=[index],marker=dict(size=16,color='#ff5544'),name=f'후보 {index+1}',hovertemplate=f'후보 {index+1} 선택<extra></extra>',showlegend=False))
    figure.update_xaxes(range=[0,image.width],visible=False)
    figure.update_yaxes(range=[image.height,0],visible=False,scaleanchor='x')
    figure.update_layout(height=500,margin=dict(l=0,r=0,t=0,b=0),clickmode='event+select',dragmode=False)
    return figure

def selected_index(event,count):
    points=event.get('selection',{}).get('points',[])
    if not points:return None
    index=points[-1].get('customdata')
    return index if isinstance(index,int) and 0<=index<count else None

def iv_curve(mobility,L,W,oxide,vt,axis,lo,hi,fixed):
    if min(mobility,L,W,oxide)<=0 or lo>=hi:raise ValueError('양수 치수/이동도와 올바른 스윕 범위가 필요합니다.')
    sweep=np.linspace(lo,hi,101)
    vg=sweep if axis=='Vg' else np.full(101,fixed)
    vd=np.full(101,fixed) if axis=='Vg' else sweep
    if (vd<0).any():raise ValueError('이 계산은 Vd ≥ 0 범위를 지원합니다.')
    overdrive=np.maximum(vg-vt,0)
    current=mobility*1e-4*cox(oxide)*W/L*np.where(vd<overdrive,overdrive*vd-.5*vd**2,.5*overdrive**2)
    return pd.DataFrame({'Vg_V':vg,'Vd_V':vd,'Id_A':current})
