import json, logging, os, time
import datetime
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexerModel, IndexToString
from pyspark.rdd import RDD
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import col, count, lit, desc
from pyspark.sql import SparkSession

#Setup system logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

def init_spark_session():
    #load the spark session(local mode)
    ss = (SparkSession.builder
         .master("local")
         .appName("Restaurant Recommendations")
         .getOrCreate())

    ss.sparkContext.setLogLevel("WARN")
    ss.sparkContext.setCheckpointDir('checkpoint')
   
    return ss

if __name__ == '__main__':
    start_time  = time.time()
    ss = init_spark_session() #initial spark session
    final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
    reviewDF = ss.read.parquet(final_indexed_save).cache()
    model_save = os.path.join('model','als_model_vegas')
    indexer_user_save = os.path.join('model','user_ind_model')

    model = ALSModel.load(model_save)
    uid = reviewDF.select('user_id').rdd.takeSample(False,1)
    logger.error('{} seconds has elapsed'.format(str(uid)))
    bid = reviewDF.select('business_id_int','business_id').distinct()
    bid.show(20)
    logger.error('{} seconds has elapsed. {} rows remain'.format(time.time() - start_time, bid.count())) 
    #predDF = bid.filter(bid['user_id'] == user_id)
    #build user request using input id
    predDF = bid.withColumn("user_id", lit(uid))
    indexer_model = StringIndexerModel.load(indexer_user_save)
    predDF = indexer_model.transform(predDF)
    '''user_id_converter =  IndexToString(inputCol= 'user_id',outputCol='user_id')
    convert_df = '''
    predDF.show(20)
    prediction_user = model.transform(predDF)
    ratings = prediction_user.sort(desc('prediction')).limit(count).select('business_id')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))