import fastapi
import pandas as pd
import numpy as np

from fastapi import HTTPException
from challenge.model import DelayModel
from datetime import datetime

app = fastapi.FastAPI()
model = DelayModel()

def get_period_day(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if(date_time > morning_min and date_time < morning_max):
            return 'mañana'
        elif(date_time > afternoon_min and date_time < afternoon_max):
            return 'tarde'
        elif(
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
        ):
            return 'noche'
    

def is_high_season(date):
    fecha_año = int(date.split('-')[0])
    date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)

    if ((date >= range1_min and date <= range1_max) or 
    (date >= range2_min and date <= range2_max) or 
    (date >= range3_min and date <= range3_max) or
    (date >= range4_min and date <= range4_max)):
        return 1
    else:
        return 0
        
def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    try:
        input_data = pd.DataFrame(data['flights'])
        preprocessed_data = model.preprocess(input_data)

        target_data = pd.read_csv(filepath_or_buffer="./data/data.csv")
        target_data['period_day'] = target_data['Fecha-I'].apply(get_period_day)
        target_data['high_season'] = target_data['Fecha-I'].apply(is_high_season)
        target_data['min_diff'] = target_data.apply(get_min_diff, axis = 1)
        target_data['delay'] = np.where(target_data['min_diff'] > 15, 1, 0)

        target_filter = (
            (target_data['OPERA'] == input_data['OPERA'][0]) &
            (target_data['TIPOVUELO'] == input_data['TIPOVUELO'][0]) &
            (target_data['MES'] == input_data['MES'][0])
        )
        target = target_data.loc[target_filter, 'delay'].values[0]

        model.fit(preprocessed_data, target)
        predictions = model.predict(preprocessed_data)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))