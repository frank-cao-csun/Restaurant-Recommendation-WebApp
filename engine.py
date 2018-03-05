import json, logging, os
import datetime
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.rdd import RDD
from pyspark.sql.context import SQLContext

#Setup system logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    
    
class RecommendationEngine:
      
    def __train_data(self):
        #Convert all Strings ids in review to int ids
        ratings = self.__convert_string_to_int()
        logger.info("Train the ALS model with current dataset")        
        self.model = ALS.train(ratings, self.rank, seed=self.seed,
                               iterations=self.iterations, lambda_=self.regularization_parameter)
        logger.info("ALS model built!")
        #Save the model to file
        now = datetime.datetime.now()
        file_name = str(now)[:16]
        model_save_path = os.path.join('model',file_name)
        self.model.save(model_save_path)
    
    @staticmethod
    def presentInList(listCategory,category):
        for i in listCategory:
            if i == category:
                return True
        return False

    def __predict_ratings(self, userId_businessId_RDD):
        #Predict rates based on ALS model(Collaborative Filtering)
        predict_ratings = self.model.predictAll(userId_businessId_RDD)
        #Convert int ids to string ids
        predict_ratings_string = self.__convert_int_to_string(predict_ratings)
        #Add user Names and business names and addresses along with predicted ratings
        user_names = self.user_ids.map(lambda x:(x[0],x[1][0]))
        business_names = self.business_ids.map(lambda x:(x[0],(x[1][0],x[1][1])))
        predict_ratings_string=predict_ratings_string.map(lambda x: (x[0],(x[1],x[2]))).join(user_names).keyBy(lambda x:x[1][0][0]).join(business_names).map(lambda x:(x[1][0][0],x[0],x[1][0][1][0][1],x[1][0][1][1],x[1][1][0],x[1][1][1])).cache()     
        print(predict_ratings_string.take(10))
        return predict_ratings_string

    def get_top_ratings(self, user_id, count):
        """Recommends up to count top unrated businesses to user_id
        """
        user_unrated_business_rdd = self.review_ids.filter(lambda rating: not rating[0] == user_id)\
                                                 .map(lambda x: (user_id, x[1])).distinct()                                                
        user_unrated_business_intids_rdd=self.__convert_string_ids_to_int(user_unrated_business_rdd)                                                                                        
        # Get predicted ratings
        ratings = self.__predict_ratings(user_unrated_business_intids_rdd)\
                                        .filter(lambda r: r[2]>=3).takeOrdered(count,key = lambda x: -x[2])    
        return ratings  
                   
    def add_ratings(self, ratings):
        """Add additional review ratings in the format (user_id, business_id, ratings)
        """
        # Convert ratings to an RDD
        new_ratings_RDD = self.sc.parallelize(ratings)
        # Add new ratings to the existing ones
        self.ratings_RDD = self.review_ids.union(new_ratings_RDD)
        # Re-train the ALS model with the new ratings
        self.__train_data()     
        return ratings 
             
    def __convert_string_ids_to_int(self,user_business_RDD):
        user_ids_replace = self.int_user_id_to_string.join(user_business_RDD).map(lambda x: (x[1][0], x[1][1]))
        requested_ids = self.int_business_id_to_string.keyBy(lambda x: x[0]).rightOuterJoin(user_ids_replace.map(lambda x: (x[1], x[0]))).map(lambda x : (x[1][1], x[1][0][1]))     
        return requested_ids 

    def __convert_int_to_string(self,user_ids_business_ids_int):
        #Convert int Ids backto string       
        user_ids_business_ids_int=user_ids_business_ids_int.map(lambda x: (x[0],(x[1],x[2])))
        #Replace userIds
        user_ids_to_string_replaced = self.reverse_mapping_user_ids.join(user_ids_business_ids_int);
        #Replace business Ids
        replace_both = user_ids_to_string_replaced.keyBy(lambda x: x[1][1][0]).join(self.reverse_mapping_business_ids).map(lambda x: (x[1][0][1][0],x[1][1],x[1][0][1][1][1]))       
        return replace_both

    def __train_all_data(self):
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
        rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        print('For testing data the RMSE is %s' % (error)            
        
    def __init__(self, spark_context, dataset):
        logger.info("Recommendation engine start")
        self.sc = spark_context
        
         # extracting review set
        review_file = os.path.join(dataset, 'yelp_academic_dataset_review.json')
        review_raw_RDD = self.sc.textFile(review_file)    
        data = review_raw_RDD.map(lambda line: json.loads(line))
        self.review_ids = data.map(lambda line: (line['user_id'], line['business_id'], line['stars'])).cache();
        
        # extract user ids and friends for social collaborative filtering
        user_file = os.path.join(dataset, 'yelp_academic_dataset_user.json')
        user_raw_RDD = self.sc.textFile(user_file)    
        user_data = user_raw_RDD.map(lambda line: json.loads(line))
        self.user_ids = user_data.map(lambda line: (line['user_id'], (line['name'],line['friends']))).cache()       
          
        # extract business_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            s
        business_file = os.path.join(dataset, 'yelp_academic_dataset_business.json')
        business_raw_RDD = self.sc.textFile(business_file)    
        business_data = business_raw_RDD.map(lambda line: json.loads(line))
        self.business_ids = business_data.map(lambda line: (line['business_id'],(line['name'], line['address'],line['categories'],line['state'],line['city'], line['latitude'],line['longitude'],line['stars'])))
        
        self.r = self.review_ids.map(lambda x: (x[0], x[2])).groupByKey()                                                                                                                                                                                                                                                                                                                                                                                              
         
        #Convert String ids to int ids and reverse it for ALS training 
        self.int_user_id_to_string = self.user_ids.map(lambda x: x[0]).distinct().zipWithUniqueId().cache()  
        self.int_business_id_to_string = self.business_ids.map(lambda x: x[0]).distinct().zipWithUniqueId().cache()   
        self.reverse_mapping_user_ids = self.int_user_id_to_string.map(lambda x: (x[1], x[0]))
        self.reverse_mapping_business_ids = self.int_business_id_to_string.map(lambda x: (x[1], x[0])) 
       
        # Train the model
        self.rank = 8
        self.seed = 5
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.__train_data()        
        logger.info("Recommendation engine Finished")