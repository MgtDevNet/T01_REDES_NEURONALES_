#!/usr/bin/env python3
"""Genera rutas DEMO sintéticas para comprobar que todo el pipeline funciona.
NO son datos reales de España y NO deben usarse en el informe final."""
from pathlib import Path
import numpy as np, pandas as pd, math
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'
cap=pd.read_excel(DATA/'capitales.xlsx')

def hav(lat1,lon1,lat2,lon2):
    R=6371.; p1,p2=np.radians([lat1,lat2]); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def make(year):
    rows=[]
    names=cap.Capital.tolist()
    for i in range(len(cap)):
        for j in range(i+1,len(cap)):
            d=float(hav(cap.Latitud.iloc[i],cap.Longitud.iloc[i],cap.Latitud.iloc[j],cap.Longitud.iloc[j]))*1.25
            # Tres alternativas artificiales: rápida, equilibrada y económica.
            for k,(speed,factor,toll) in enumerate([(88,1.00,0),(78,0.98,8),(70,0.97,0)]):
                dist=d*factor
                # pequeñas diferencias deterministas entre escenarios
                if year==2020: speed*=0.96; toll*=0.90
                t=dist/speed
                # Peaje sintético solo para la ruta 2; NO es tarifa real.
                rows.append({'origen':names[i],'destino':names[j],'route_id':f'{i:02d}_{j:02d}_R{k+1}', 'distance_km':round(dist,3),'time_h':round(t,4),'toll_eur':float(toll),'route_type':['rapida_demo','mixta_demo','economica_demo'][k], 'route_description':'DATOS SINTÉTICOS DE DEMOSTRACIÓN','source':'DEMO: generado localmente','source_date':'2026-10-08'})
    pd.DataFrame(rows).to_csv(DATA/f'rutas_{year}.csv',index=False,encoding='utf-8-sig')
for y in [2020,2026]: make(y)
print('Generadas rutas_2020.csv y rutas_2026.csv (DEMO sintético).')
