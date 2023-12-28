from collections import defaultdict

from flask import Flask, request, jsonify, render_template
import math
import time
import random
from nltk.internals import Counter
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from pydocumentdb import document_client

app = Flask(__name__)
ENDPOINT = "https://tutorial-uta-cse6332.documents.azure.com:443/"
MASTERKEY = "fSDt8pk5P1EH0NlvfiolgZF332ILOkKhMdLY6iMS2yjVqdpWx4XtnVgBoJBCBaHA8PIHnAbFY4N9ACDbMdwaEw=="
DATABASE_ID = "tutorial"
COLLECTION_ID1 = "us_cities"
COLLECTION_ID2 = "reviews"
client = document_client.DocumentClient(ENDPOINT, {'masterKey': MASTERKEY})


def get_cities_data():
    sql = "SELECT c.city, c.lat, c.lng, c.population FROM c"
    o = {"enableCrossPartitionQuery": True}
    r = list(client.QueryDocuments(f"dbs/{DATABASE_ID}/colls/{COLLECTION_ID1}", sql, o))
    c = []
    for i in r:
        c.append({
            "city": i['city'],
            "lat": i['lat'],
            "lng": i['lng'],
            "population": i["population"]
        })
    return c


def get_reviews_data():
    # sql = "SELECT c.score, c.city FROM c"
    sql = "SELECT TOP 100 c.score, c.city FROM c"
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


# 计算欧几里得距离
def calculate_euclidean_distance(city1, city2):
    lat1, lon1 = city1['lat'], city1['lng']
    lat2, lon2 = city2['lat'], city2['lng']
    return math.sqrt((float(lat1) - float(lat2)) ** 2 + (float(lon1) - float(lon2)) ** 2)


# KNN算法实现
def knn_clustering(classes, k, words):
    # 初始化聚类
    clusters = {i: [] for i in range(classes)}

    cities_data = get_cities_data()

    # 为每个城市分配一个初始类别
    for city in cities_data:
        assigned_class = random.randint(0, classes - 1)
        clusters[assigned_class].append(city)

    # 迭代分配城市到最近的类别
    for city in cities_data:
        distances = []
        for class_id in clusters:
            center = clusters[class_id][0]  # 假设每个类别的第一个城市是中心
            distance = calculate_euclidean_distance(city, center)
            distances.append((class_id, distance))
        distances.sort(key=lambda x: x[1])
        nearest_classes = [class_id for class_id, _ in distances[:k]]
        most_common_class = Counter(nearest_classes).most_common(1)[0][0]
        clusters[most_common_class].append(city)

    return clusters


@app.route('/stat/knn_reviews', methods=['GET'])
def knn_reviews():
    start_time = time.time()

    # 从请求中获取参数
    classes = int(request.args.get('classes', 6))
    k = int(request.args.get('k', 3))
    words = int(request.args.get('words', 100))

    # 执行KNN聚类
    clusters = knn_clustering(classes, k, words)

    # 计算每个类的总人口
    cluster_populations = {class_id: sum(int(city['population']) for city in cities) for class_id, cities in
                           clusters.items()}

    # 计算响应时间
    elapsed_time = (time.time() - start_time) * 1000

    response = {
        'clusters': [{'classId': class_id, 'population': population} for class_id, population in
                     cluster_populations.items()],
        'time_ms': elapsed_time
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
