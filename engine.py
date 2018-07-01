import json, logging, os, time
import pandas
from pyfiglet import figlet_format
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexerModel, IndexToString
from pyspark.rdd import RDD
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import col, count, lit, desc

#Setup system logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    
    
class RecommendationEngine:
      
    def __init__(self, spark_session, dataset):
        logger.warn(figlet_format('Eat-Smart',font='big'))
        logger.error("Eat-Smart core start")    
        self.ss = spark_session
        self.ds = dataset
        # Train the model
        self.test_api()
        #self.train_model()        
        logger.error("Eat-Smart Core Ready")

    def test_api(self):

        final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
        self.reviewDF = self.ss.read.parquet(final_indexed_save).cache()
        self.model_save = os.path.join('model','als_model_vegas')
        self.indexer_user_save = os.path.join('model','user_ind_model')
    
    def train_model(self):
        #load als
        final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
        self.reviewDF = self.ss.read.parquet(final_indexed_save).cache()
        logger.error("Train the ALS model with current dataset")        
        als = ALS(rank= 8, maxIter= 15, regParam=0.25, userCol="user_id_num", 
              itemCol="business_id_num", ratingCol="stars_long", coldStartStrategy="nan")
        self.model = als.fit(self.reviewDF)
        logger.error("ALS model built!")
        #Save the model to file
        self.model_save = os.path.join('model','als_model_vegas')
        self.model.write().overwrite().save(self.model_save)
        #now = datetime.datetime.now()
        #file_name = str(now)[:16]


    def __predict_ratings(self, predDF):
        #Predict rates based on ALS model(Collaborative Filtering)
        #predict_ratings = self.model.predictAll(userId_businessId_RDD)
        #Convert int ids to string ids
       
        #Add user Names and business names and addresses along with predicted ratings
        user_names = self.user_ids.map(lambda x:(x[0],x[1][0]))
        business_names = self.business_ids.map(lambda x:(x[0],(x[1][0],x[1][1])))
        predict_ratings_string=predict_ratings_string.map(lambda x: (x[0],(x[1],x[2]))).join(user_names).keyBy(lambda x:x[1][0][0]).join(business_names).map(lambda x:(x[1][0][0],x[0],x[1][0][1][0][1],x[1][0][1][1],x[1][1][0],x[1][1][1])).cache()     
        print(predict_ratings_string.take(10))
        return predict_ratings_string

    def get_top_ratings(self, user_id, count):
        """Retrun top <count> bussiness
           Calls 
        """
        start_time  = time.time()
        #bid = self.reviewDF.select('business_id_num','business_id').distinct().cache()
        businessDF_vegas_food_save = os.path.join('dataset','businessDF_vegas_food.parquet')
        businessDF_vegas_food = self.ss.read.parquet(businessDF_vegas_food_save)

        #bid.show(20)
        logger.error('{} seconds has elapsed. {} entries remained'.format(time.time() - start_time, businessDF_vegas_food.count())) 
        #predDF = bid.filter(bid['user_id'] == user_id)
        #build user request using input id
        logger.error('{} seconds has elapsed before loading building predDF'.format(time.time() - start_time))
        bid = businessDF_vegas_food.select('business_id','latitude','longitude')
        indexer_business_save = os.path.join('model', 'bus_ind_model')
        indexer_business_model = StringIndexerModel.load(indexer_business_save)
        bid = indexer_business_model.transform(bid)
        predDF = bid.withColumn("user_id", lit(user_id)).cache()

        logger.error('{} seconds has elapsed before loading indexer'.format(time.time() - start_time))
        indexer_user_model = StringIndexerModel.load(self.indexer_user_save)
        predDF = indexer_user_model.transform(predDF)
        '''user_id_converter =  IndexToString(inputCol= 'user_id',outputCol='user_id')
        convert_df = '''
        #predDF.show(10)
        logger.error('{} seconds has elapsed before model'.format(time.time() - start_time))
        model = ALSModel.load(self.model_save)
        prediction_user = model.transform(predDF)
        #prediction_user.show(20)
        ratings = prediction_user.sort(desc('prediction')).limit(count).select('business_id','prediction','latitude','longitude')
        #ratings.show(20)
        #ratings.printSchema()
        logger.error('{} seconds has elapsed'.format(time.time() - start_time))
        return ratings.toPandas().to_json(orient='records')
                   
    def add_ratings(self, ratings):
        """Add additional review ratings in the format (user_id, business_id, ratings)
        """
        # Convert ratings to an RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # Add new ratings to the existing ones
        self.ratings_RDD = self.review_ids.union(new_ratings_RDD)
        # Re-train the ALS model with the new ratings
        #self.__train_data()     
        return ratings

    @staticmethod
    def presentInList(listCategory,category):
        for i in listCategory:
            if i == category:
                return True
        return False 

    
    def train_best_model(self):

        min_error = float('inf') 
        best_rank = -1
        best_iteration = -1
        ranks = [4,8,12]
        errors = [0, 0, 0]
        err = 0
        tolerance = 0.02
        min_error = float('inf')
        best_rank = -1
        best_iteration = -1
        #Convert all Strings ids in review to int ids
        ratings = self.__convert_string_to_int()
        training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2], seed=0)
        validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
        test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
        
        for rank in ranks:
            model = ALS.train(training_RDD, rank, seed=self.seed, iterations=self.iterations,
                      lambda_=self.regularization_parameter)
            predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
            rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
            error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
            errors[err] = error
            err += 1
            print('For rank %s the RMSE is %s' % (rank, error))
            if error < min_error:
                min_error = error
                best_rank = rank

        print('The best model was trained with rank %s' % best_rank)
        
        predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        #rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        #error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        #print('For testing data the RMSE is %s' % (error)