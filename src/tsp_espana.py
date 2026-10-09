#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TSP España: 47 capitales peninsulares.

Motor reproducible para el trabajo. No consulta Internet durante la ejecución.
Los CSV de rutas deben contener rutas candidatas previamente documentadas.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; RESULTS=ROOT/'results'
CAP=DATA/'capitales.xlsx'; P20=DATA/'rutas_2020.csv'; P26=DATA/'rutas_2026.csv'; PARAM=DATA/'parametros.json'
REQ_CITY={'ID','Provincia','Capital','Latitud','Longitud'}
REQ_ROUTE={'origen','destino','route_id','distance_km','time_h','toll_eur'}
SEED=20261008

def norm(x): return ' '.join(str(x).strip().split()).lower()
def pair(a,b): return tuple(sorted((norm(a),norm(b))))
def seed_all(s): random.seed(s); np.random.seed(s)
def load_capitals():
    df=pd.read_excel(CAP)
    miss=REQ_CITY-set(df.columns)
    if miss: raise ValueError(f'Faltan columnas en capitales.xlsx: {sorted(miss)}')
    if len(df)!=47: raise ValueError(f'Se esperaban 47 capitales, hay {len(df)}')
    if df.Capital.duplicated().any(): raise ValueError('Hay capitales duplicadas.')
    if df[['Latitud','Longitud']].isna().any().any(): raise ValueError('Hay coordenadas faltantes.')
    return df.reset_index(drop=True)

def load_routes(path,capitals,scenario):
    if not path.exists(): raise FileNotFoundError(f'Falta {path}')
    df=pd.read_csv(path)
    miss=REQ_ROUTE-set(df.columns)
    if miss: raise ValueError(f'Faltan columnas en {path.name}: {sorted(miss)}')
    for c in ['route_type','route_description','source','source_date']:
        if c not in df: df[c]=''
    cities=set(capitals.Capital)
    unknown=(set(df.origen)|set(df.destino))-cities
    if unknown: raise ValueError(f'Ciudades desconocidas: {sorted(unknown)}')
    for c in ['distance_km','time_h','toll_eur']: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[['distance_km','time_h','toll_eur']].isna().any().any(): raise ValueError(f'Valores faltantes/no numéricos en {path.name}')
    if (df.distance_km<=0).any() or (df.time_h<=0).any() or (df.toll_eur<0).any(): raise ValueError(f'Valores inválidos en {path.name}')
    df['pair']=[pair(a,b) for a,b in zip(df.origen,df.destino)]; df['scenario']=scenario
    return df

def load_params():
    with open(PARAM,encoding='utf-8') as f: return json.load(f)

def cost_matrix(routes,capitals,vh,fuel_price,consumption):
    names=capitals.Capital.tolist(); idx={norm(x):i for i,x in enumerate(names)}; n=len(names)
    t=routes.copy(); fuel_km=consumption/100*fuel_price
    t['fuel_eur']=t.distance_km*fuel_km; t['time_cost_eur']=vh*t.time_h
    t['total_cost_eur']=t.fuel_eur+t.toll_eur+t.time_cost_eur
    C=np.full((n,n),np.inf); selected=[]
    for p,g in t.groupby('pair',sort=False):
        r=g.loc[g.total_cost_eur.idxmin()]; a,b=p; i,j=idx[a],idx[b]; C[i,j]=C[j,i]=r.total_cost_eur
        selected.append(r.to_dict())
    np.fill_diagonal(C,0)
    if not np.isfinite(C).all():
        miss=[]
        for i in range(n):
            for j in range(i+1,n):
                if not np.isfinite(C[i,j]): miss.append((names[i],names[j]))
        raise ValueError(f'Faltan {len(miss)} pares de rutas. Ejemplos: {miss[:5]}')
    return C,pd.DataFrame(selected)

def tour_cost(r,C): return float(sum(C[r[k],r[(k+1)%len(r)]] for k in range(len(r))))
def canon(r):
    r=np.asarray(r,dtype=int); r=np.roll(r,-int(np.where(r==0)[0][0]));
    if r[1]>r[-1]: r=np.r_[0,r[:0:-1]]
    return r

def aco(C,cfg,seed):
    rng=np.random.default_rng(seed); n=len(C); ants=int(cfg['ants']); iters=int(cfg['iterations'])
    alpha=float(cfg['alpha']); beta=float(cfg['beta']); rho=float(cfg['rho']); q=float(cfg['q']); ew=float(cfg['elite_weight'])
    eta=np.zeros_like(C); m=C>0; eta[m]=1/C[m]; tau=np.ones_like(C); np.fill_diagonal(tau,0)
    best=None; bestc=math.inf; hist=[]; rh=[]
    for it in range(iters):
        routes=[]; costs=[]
        for _ in range(ants):
            r=[0]; un=set(range(1,n))
            while un:
                cur=r[-1]; cand=np.array(sorted(un)); w=(tau[cur,cand]**alpha)*(eta[cur,cand]**beta)
                p=np.ones(len(cand))/len(cand) if (not np.isfinite(w).all() or w.sum()<=0) else w/w.sum()
                nxt=int(rng.choice(cand,p=p)); r.append(nxt); un.remove(nxt)
            r=np.array(r); c=tour_cost(r,C); routes.append(r); costs.append(c)
            if c<bestc: bestc=c; best=r.copy()
        tau*=1-rho
        for r,c in zip(routes,costs):
            d=q/max(c,1e-12)
            for k in range(n): a,b=r[k],r[(k+1)%n]; tau[a,b]+=d; tau[b,a]+=d
        d=q*ew/max(bestc,1e-12)
        for k in range(n): a,b=best[k],best[(k+1)%n]; tau[a,b]+=d; tau[b,a]+=d
        hist.append((it+1,bestc,min(costs))); rh.append(best.copy())
    return canon(best),bestc,pd.DataFrame(hist,columns=['iteration','best_cost','iteration_best']),rh

def random_route(n,rng):
    r=np.arange(n); x=r[1:].copy(); rng.shuffle(x); r[1:]=x; return r

def ox(p1,p2,rng):
    n=len(p1); a,b=sorted(rng.choice(np.arange(1,n),2,replace=False))
    def child(x,y):
        c=np.full(n,-1); c[0]=0; c[a:b]=x[a:b]; used=set(c[a:b])|{0}; pos=b
        for g in np.r_[y[b:],y[1:b]]:
            if g in used: continue
            if pos>=n: pos=1
            c[pos]=g; used.add(int(g)); pos+=1
        return c
    return child(p1,p2),child(p2,p1)

def mutate(r,rng):
    c=r.copy(); a,b=sorted(rng.choice(np.arange(1,len(r)),2,replace=False)); c[a:b]=c[a:b][::-1]; return c

def ga(C,cfg,seed):
    rng=np.random.default_rng(seed); n=len(C); N=int(cfg['population']); G=int(cfg['generations']); elite=int(cfg['elite']); tk=int(cfg['tournament_k']); cr=float(cfg['crossover_rate']); mr=float(cfg['mutation_rate'])
    pop=[random_route(n,rng) for _ in range(N)]; best=None; bestc=math.inf; hist=[]; rh=[]
    for g in range(G):
        costs=np.array([tour_cost(r,C) for r in pop]); order=np.argsort(costs); pop=[pop[i] for i in order]; costs=costs[order]
        if costs[0]<bestc: bestc=float(costs[0]); best=pop[0].copy()
        hist.append((g+1,bestc,float(costs[0]),float(costs.mean()))); rh.append(best.copy())
        new=[pop[i].copy() for i in range(min(elite,N))]
        while len(new)<N:
            ix=rng.choice(N,tk,replace=False); p1=pop[ix[np.argmin(costs[ix])]].copy(); ix=rng.choice(N,tk,replace=False); p2=pop[ix[np.argmin(costs[ix])]].copy()
            c1,c2=ox(p1,p2,rng) if rng.random()<cr else (p1,p2)
            if rng.random()<mr: c1=mutate(c1,rng)
            if rng.random()<mr and len(new)+1<N: c2=mutate(c2,rng)
            new.append(c1)
            if len(new)<N: new.append(c2)
        pop=new
    return canon(best),bestc,pd.DataFrame(hist,columns=['generation','best_cost','generation_best','mean_cost']),rh

def names(route,cap): return [cap.Capital.iloc[i] for i in route]+[cap.Capital.iloc[route[0]]]
def save_route(r,c,cap,alg,sc,vh,out):
    pd.DataFrame({'orden':range(1,len(r)+2),'capital':names(r,cap)}).to_csv(out,index=False,encoding='utf-8-sig')
    return {'scenario':sc,'algorithm':alg,'hour_value_eur_h':vh,'total_cost_eur':c,'tour':' -> '.join(names(r,cap))}

def conv(h,x,title,out):
    plt.figure(figsize=(9,5)); plt.plot(h[x],h.best_cost); plt.xlabel(x); plt.ylabel('Mejor costo (€)'); plt.title(title); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(out,dpi=150); plt.close()
def route_map(r,cap,title,out):
    d=cap.iloc[r].copy(); d=pd.concat([d,d.iloc[[0]]]); plt.figure(figsize=(9,8)); plt.plot(d.Longitud,d.Latitud,'-o',ms=3); 
    for _,z in d.iloc[:-1].iterrows(): plt.text(z.Longitud,z.Latitud,z.Capital,fontsize=5)
    plt.xlabel('Longitud'); plt.ylabel('Latitud'); plt.title(title); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
def gif(hist,cap,title,out,maxf=60):
    if len(hist)>maxf: hist=[hist[i] for i in np.linspace(0,len(hist)-1,maxf).astype(int)]
    lon=cap.Longitud.to_numpy(); lat=cap.Latitud.to_numpy(); fig,ax=plt.subplots(figsize=(8,7)); ax.scatter(lon,lat,s=15); [ax.text(lon[i],lat[i],cap.Capital.iloc[i],fontsize=4) for i in range(len(cap))]; line,=ax.plot([],[],'-o',ms=2); txt=ax.text(.02,.98,'',transform=ax.transAxes,va='top'); ax.set_title(title); ax.set_xlabel('Longitud'); ax.set_ylabel('Latitud')
    def up(k):
        r=canon(hist[k]); q=np.r_[r,r[0]]; line.set_data(lon[q],lat[q]); txt.set_text(f'Frame {k+1}/{len(hist)}'); return line,txt
    a=FuncAnimation(fig,up,frames=len(hist),interval=100,blit=True); a.save(out,writer=PillowWriter(fps=8)); plt.close(fig)

def run(sc,routes,cap,p,vh,fp,seed):
    C,sel=cost_matrix(routes,cap,vh,fp,p['vehicle']['consumption_l_100km']); out=RESULTS/sc/f'vh_{vh:g}'; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(C,index=cap.Capital,columns=cap.Capital).to_csv(out/'matriz_costos.csv',encoding='utf-8-sig'); sel.to_csv(out/'rutas_seleccionadas.csv',index=False,encoding='utf-8-sig')
    ar,ac,ah,arh=aco(C,p['aco'],seed); gr,gc,gh,grh=ga(C,p['ga'],seed+1)
    ah.to_csv(out/'historial_aco.csv',index=False); gh.to_csv(out/'historial_ga.csv',index=False)
    a=save_route(ar,ac,cap,'ACO',sc,vh,out/'ruta_aco.csv'); g=save_route(gr,gc,cap,'GA',sc,vh,out/'ruta_ga.csv')
    conv(ah,'iteration',f'ACO {sc} Vh={vh:g}',out/'convergencia_aco.png'); conv(gh,'generation',f'GA {sc} Vh={vh:g}',out/'convergencia_ga.png')
    route_map(ar,cap,f'Ruta ACO - {sc}',out/'mapa_aco.png'); route_map(gr,cap,f'Ruta GA - {sc}',out/'mapa_ga.png'); gif(arh,cap,f'Evolución ACO - {sc}',out/'evolucion_aco.gif'); gif(grh,cap,f'Evolución GA - {sc}',out/'evolucion_ga.gif')
    return [a,g]

def sensitivity(sc,routes,cap,p,fp,seed):
    s=p['sensitivity']; vals=np.arange(s['min'],s['max']+s['step']/2,s['step']); rec=[]
    for vh in vals:
        C,_=cost_matrix(routes,cap,float(vh),fp,p['vehicle']['consumption_l_100km']); ar,ac,_,_=aco(C,p['aco'],seed); gr,gc,_,_=ga(C,p['ga'],seed+1)
        rec += [{'scenario':sc,'algorithm':'ACO','hour_value_eur_h':vh,'total_cost_eur':ac,'tour_key':'-'.join(map(str,canon(ar)))},{'scenario':sc,'algorithm':'GA','hour_value_eur_h':vh,'total_cost_eur':gc,'tour_key':'-'.join(map(str,canon(gr)))}]
    d=pd.DataFrame(rec); d.to_csv(RESULTS/f'sensibilidad_{sc}.csv',index=False,encoding='utf-8-sig')
    plt.figure(figsize=(9,5));
    for a in ['ACO','GA']:
        q=d[d.algorithm==a]; plt.plot(q.hour_value_eur_h,q.total_cost_eur,label=a)
    plt.xlabel('Valor hora (€/h)'); plt.ylabel('Costo óptimo (€)'); plt.title(f'Sensibilidad {sc}'); plt.legend(); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(RESULTS/f'sensibilidad_{sc}.png',dpi=150); plt.close(); return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',choices=['2020','2026','both'],default='both'); ap.add_argument('--hour-value',type=float); ap.add_argument('--seed',type=int); ap.add_argument('--sensitivity',action='store_true'); args=ap.parse_args()
    RESULTS.mkdir(exist_ok=True); cap=load_capitals(); p=load_params(); seed=p['seed'] if args.seed is None else args.seed
    cons=p['vehicle']['consumption_l_100km']; assert cons>0
    todo=[]
    if args.scenario in ('2020','both'): todo.append(('2020',load_routes(P20,cap,'2020')))
    if args.scenario in ('2026','both'): todo.append(('2026',load_routes(P26,cap,'2026')))
    summary=[]
    for sc,r in todo:
        fp=p['fuel_prices_eur_l'][sc]; vh=args.hour_value if args.hour_value is not None else p['hour_value_eur'][f'base_{sc}']; summary+=run(sc,r,cap,p,float(vh),float(fp),seed)
        if args.sensitivity: sensitivity(sc,r,cap,p,float(fp),seed)
    sm=pd.DataFrame(summary); sm.to_csv(RESULTS/'resumen_principal.csv',index=False,encoding='utf-8-sig')
    if len(todo)==2:
        piv=sm.pivot_table(index='algorithm',columns='scenario',values='total_cost_eur',aggfunc='first').reset_index(); piv['difference_2026_minus_2020_eur']=piv['2026']-piv['2020']; piv['difference_percent']=100*piv['difference_2026_minus_2020_eur']/piv['2020']; piv.to_csv(RESULTS/'contraste_2020_2026.csv',index=False,encoding='utf-8-sig')
    print('\n=== TSP ESPAÑA: EJECUCIÓN COMPLETA ==='); print(f'Capitales: {len(cap)} | semilla: {seed}'); print(sm[['scenario','algorithm','hour_value_eur_h','total_cost_eur']].to_string(index=False)); print(f'Resultados: {RESULTS.resolve()}')
if __name__=='__main__': main()
