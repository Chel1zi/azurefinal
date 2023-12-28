from collections import defaultdict

from flask import Flask, request, jsonify, render_template
import math
import time
from nltk.internals import Counter
from nltk.tokenize import word_tokenize

from pydocumentdb import document_client

app = Flask(__name__)
ENDPOINT = "https://tutorial-uta-cse6332.documents.azure.com:443/"
MASTERKEY = "fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
DATABASE_ID = "tutorial"
COLLECTION_ID1 = "us_cities"
COLLECTION_ID2 = "reviews"
client = document_client.DocumentClient(ENDPOINT, {'masterKey': MASTERKEY})


def get_cities_data():
    sql = "SELECT c.city, c.lat, c.lng FROM c"
    o = {"enableCrossPartitionQuery": True}
    r = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID1}", sql, o))
    c = []
    for i in r:
        c.append({
            "city": i['city'],
            "lat": i['lat'],
            "lng": i['lng']
        })
    return c


def get_reviews_data():
    # sql = "SELECT c.score, c.city FROM c"
    sql = "SELECT TOP 1000 c.score, c.city FROM c"
    o = {"enableCrossPartitionQuery": True}  # 如果集合是分区集合，需要启用跨分区查询
    # 执行查询
    q = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID2}", sql, o))
    d = []
    for i in q:
        d.append({
            "score": i['score'],
            "city": i['city'],
        })
    return d


# Helper function to calculate Eular distance
def calculate_eular_distance(lat1, lng1, lat2, lng2):
    return math.sqrt((lat1 - lat2) ** 2 + (lng1 - lng2) ** 2)


# Route to handle closest cities query
@app.route('/stat/closest_cities', methods=['GET'])
def closest_cities():
    start_time = time.time()

    city_name = request.args.get('city')
    page_size = int(request.args.get('page_size', 50))
    page = int(request.args.get('page', 0))

    cities_data = get_cities_data()

    # Find the requested city's coordinates
    requested_city = next((city for city in cities_data if city['city'].lower() == city_name.lower()), None)
    if not requested_city:
        return jsonify({"error": "City not found"}), 404

    # Calculate distances
    distances = []
    for city in cities_data:
        if city['city'].lower() != city_name.lower():
            distance = calculate_eular_distance(float(requested_city['lat']), float(requested_city['lng']),
                                                float(city['lat']), float(city['lng']))
            distances.append((city['city'], distance))

    # Sort by distance
    sorted_cities = sorted(distances, key=lambda x: x[1])

    # Implement pagination
    start = page * page_size
    end = start + page_size
    paginated_cities = sorted_cities[start:end]

    # Compute response time
    elapsed_time = (time.time() - start_time) * 1000  # Time in milliseconds

    return jsonify({
        "closest_cities": paginated_cities,
        "time_ms": elapsed_time
    })


@app.route('/stat/closest_cities_reviews', methods=['GET'])
def closest_cities_reviews():
    print("start")
    start_time = time.time()

    city_name = request.args.get('city')
    page_size = int(request.args.get('page_size', 50))
    page = int(request.args.get('page', 0))

    cities_data = get_cities_data()
    reviews_data = get_reviews_data()

    print("first step")
    # 预先计算所有城市的平均评分
    city_scores = {}
    for review in reviews_data:
        city_lower = review['city'].lower()
        if city_lower in city_scores:
            city_scores[city_lower].append(review['score'])
        else:
            city_scores[city_lower] = [review['score']]

    # avg_scores = {city: sum(scores) / len(scores) if scores else 0 for city, scores in city_scores.items()}
    avg_scores = {city: sum(float(score) for score in scores) / len(scores) if scores else 0 for city, scores in
                  city_scores.items()}

    print("second step")
    # Find the requested city's coordinates
    requested_city = next((city for city in cities_data if city['city'].lower() == city_name.lower()), None)
    if not requested_city:
        return jsonify({"error": "City not found"}), 404

    distances = []
    for city in cities_data:
        if city['city'].lower() != city_name.lower():
            distance = calculate_eular_distance(float(requested_city['lat']), float(requested_city['lng']),
                                                float(city['lat']), float(city['lng']))
            avg_score = avg_scores.get(city['city'].lower(), 0)
            distances.append((city['city'], distance, avg_score))
    print("third step")
    # Sort by distance
    sorted_cities = sorted(distances, key=lambda x: x[1])

    # Implement pagination
    start = page * page_size
    end = start + page_size
    paginated_cities = sorted_cities[start:end]

    # Compute response time
    elapsed_time = (time.time() - start_time) * 1000  # Time in milliseconds

    return jsonify({
        "closest_cities": paginated_cities,
        "time_ms": elapsed_time
    })


@app.route('/stat', methods=['GET'])
def stat():
    return render_template('index.html')


# 从文件读取停用词
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    return stopwords


stop_words = load_stopwords('./static/stopwords.txt')


# 辅助函数：从评论中提取词汇
def extract_words(reviews):
    words = []
    for review in reviews:
        word_tokens = word_tokenize(review.lower())
        filtered_words = [word for word in word_tokens if word.isalpha() and word not in stop_words]
        words.extend(filtered_words)
    return words


@app.route('/stat/knn_reviews', methods=['GET'])
def knn_reviews():
    start_time = time.time()

    # 从请求中获取参数
    classes = int(request.args.get('classes', 6))
    k = int(request.args.get('k', 3))
    words_count = int(request.args.get('words', 100))

    # 从数据库获取评论数据
    reviews_data = get_reviews_data()
    class_vectors = {}
    for class_id in range(classes):
        class_reviews = [review['review'] for review in reviews_data if review['class_id'] == class_id]
        class_vectors[class_id] = extract_words(class_reviews)
    elapsed_time = (time.time() - start_time) * 1000
    class_words = defaultdict(list)
    for class_id in range(classes):
        class_reviews = [review['review'] for review in reviews_data if review['class_id'] == class_id]
        all_words = extract_words(class_reviews)
        word_freq = Counter(all_words)
        most_common = word_freq.most_common(words_count)
        class_words[class_id] = most_common
    # 计算响应时间
    elapsed_time = (time.time() - start_time) * 1000
    # 构建并返回 JSON 响应
    response = {
        "class_words": class_words,
        "time_ms": elapsed_time
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
