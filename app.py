import engine
from engine import RecommendationEngine
from flask import Blueprint
from flask.app import Flask
from cherrypy._cpcompat import json
from asyncio.log import logger
from flask import Flask, request,render_template
main = Blueprint('main',__name__)


@main.route("/<string:user_id>/ratings/top/<int:count>", methods=["GET"])
def top_recommendation(user_id, count):
    """Get Top Recommendations """
    top_ratings = recommendationengine.get_top_ratings(user_id, count);
    return top_ratings

@main.route("/", methods=["GET"])
def mainIndex():
    logger.debug("User %s rating requested for movie %s")
    return render_template('index.html')

@main.route("/<string:user_id>/ratings", methods = ["POST"])
def add_ratings(user_id):
    # get the ratings from the Flask POST request object http://<SERVER_IP>:5432/0/ratings/top/10
    ratings_list = request.form.keys()[0].strip().split("\n")
    ratings_list = map(lambda x: x.split(","), ratings_list)
    # create a list with the format required by the negine (user_id, movie_id, rating)
    ratings = map(lambda x: (user_id, int(x[0]), float(x[1])), ratings_list)
    # add them to the model using then engine API
    recommendationengine.add_ratings(ratings)
 
    return json.dumps(ratings)

def create_app(spark_session,dataset):
    global recommendationengine
    recommendationengine = RecommendationEngine(spark_session,dataset)  
    app = Flask(__name__)
    app.register_blueprint(main)
    return app 
