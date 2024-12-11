import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

    
# 데이터 준비
ratings_file = "/home/work/yunyeong/정제/ratings.dat"
users_file = "/home/work/yunyeong/정제/users.dat"
subgenre_file = "/home/work/yunyeong/정제/subgenre.csv"
merged_file = "/home/work/yunyeong/정제/merged_cleaned.csv"
fasttext_files = [
    "/home/work/yunyeong/정제/wiki.en.part0.vec",
    "/home/work/yunyeong/정제/wiki.en.part1.vec",
    "/home/work/yunyeong/정제/wiki.en.part2.vec"
]

# 데이터 로드: MovieLens와 merged 파일
ratings = pd.read_csv(ratings_file, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')
users = pd.read_csv(users_file, sep='::', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], engine='python')
movies = pd.read_csv(merged_file, usecols=['MovieID', 'Title', 'Genres', 'overview'], engine='python')
subgenres_df = pd.read_csv(subgenre_file)

# Genre - Subgenre 매핑
genre_subgenre_map = {}
for g, sg in subgenres_df.values:
    if g not in genre_subgenre_map:
        genre_subgenre_map[g] = []
    genre_subgenre_map[g].append(sg)

# Genres를 리스트 형태로 변환
movies['Genres_list'] = movies['Genres'].str.split('|')

# 텍스트 전처리
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

movies['Title'] = movies['Title'].apply(preprocess_text)
movies['overview'] = movies['overview'].apply(preprocess_text)



# FastText 벡터 로드 함수
def load_fasttext_vectors(files):
    word_vectors = {}
    header_processed = False
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if not header_processed:
                    # 첫 줄은 헤더
                    header_processed = True
                    continue
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                vector = np.array(tokens[1:], dtype=float)
                word_vectors[word] = vector
    return word_vectors

# FastText 임베딩 로드
fasttext_vectors = load_fasttext_vectors(fasttext_files)
embedding_dim = len(next(iter(fasttext_vectors.values())))

# 텍스트 임베딩 함수 (Genre/Subgenre 임베딩에도 재사용)
def get_embedding(text, word_vectors, embedding_dim):
    if not isinstance(text, str):
        return np.zeros(embedding_dim)
    tokens = text.split()
    vecs = [word_vectors[word] for word in tokens if word in word_vectors]
    if len(vecs) == 0:
        return np.zeros(embedding_dim)
    return np.mean(vecs, axis=0)


# Subgenre 텍스트에 대한 임베딩 계산
# Subgenre 이름 자체를 전처리 후 임베딩 벡터 생성
subgenre_set = set([sg for sgs in genre_subgenre_map.values() for sg in sgs])
subgenre_embeddings = {}
for sg in subgenre_set:
    sg_processed = preprocess_text(sg)
    subgenre_embeddings[sg] = get_embedding(sg_processed, fasttext_vectors, embedding_dim)


# 영화 벡터 생성 함수: Title + Overview 평균
def get_movie_vector(title, overview, word_vectors, embedding_dim):
    title_vec = get_embedding(title, word_vectors, embedding_dim)
    overview_vec = get_embedding(overview, word_vectors, embedding_dim)
    return (title_vec + overview_vec) / 2.0

# 각 영화별 기본 임베딩(Title+Overview) 계산
movie_base_vector_cache = {
    movie_row['MovieID']: get_movie_vector(
        movie_row['Title'], movie_row['overview'], fasttext_vectors, embedding_dim
    )
    for _, movie_row in movies.iterrows()
}

# 장르 기반 Subgenre 후보 및 최적 Subgenre 할당
# - 각 영화의 Genres_list를 통해 가능한 Subgenre 후보를 가져옴
# - 영화 벡터와 각 Subgenre 벡터 사이의 코사인 유사도를 계산하여 최적 Subgenre 선택
movie_final_vector_cache = {}
for _, row in movies.iterrows():
    movie_id = row['MovieID']
    movie_vec = movie_base_vector_cache[movie_id]
    
    # 영화가 속한 모든 Genre에 대한 Subgenre 후보 수집
    candidate_subgenres = []
    for g in row['Genres_list']:
        if g in genre_subgenre_map:
            candidate_subgenres.extend(genre_subgenre_map[g])
    
    # 후보 Subgenre가 없다면, 그냥 영화 벡터만 사용
    if len(candidate_subgenres) == 0:
        movie_final_vector_cache[movie_id] = movie_vec
        continue
    
    # 코사인 유사도로 최적 Subgenre 선택 (가장 유사도 높은 Subgenre 선택)
    candidate_vectors = np.array([subgenre_embeddings[sg] for sg in candidate_subgenres])
    movie_vec_reshaped = movie_vec.reshape(1, -1)
    similarities = cosine_similarity(movie_vec_reshaped, candidate_vectors)[0]
    best_index = np.argmax(similarities)
    best_subgenre = candidate_subgenres[best_index]
    best_subgenre_vec = subgenre_embeddings[best_subgenre]
    
    # 영화벡터와 서브장르벡터를 결합하는 방식 결정
    # 단순 평균 벡터를 영화의 최종 벡터로 사용
    movie_final_vec = (movie_vec + best_subgenre_vec) / 2.0
    movie_final_vector_cache[movie_id] = movie_final_vec


# 사용자 취향 벡터 구성
# 사용자 선호(평점>=4) 영화들의 최종 영화벡터(영화+Subgenre 포함) 평균을 취함
def get_user_preference_vector(user_ratings, movie_vector_cache, embedding_dim):
    liked_movies = user_ratings[user_ratings['Rating'] >= 4]
    vectors = [movie_vector_cache.get(mid, np.zeros(embedding_dim)) for mid in liked_movies['MovieID']]
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

user_preference_cache = {
    user_id: get_user_preference_vector(
        ratings[ratings['UserID'] == user_id],
        movie_final_vector_cache,
        embedding_dim
    )
    for user_id in ratings['UserID'].unique()
}

# 유저 특성 활용: 나이(Age), 성별(Gender) 반영
# Gender를 M=1, F=0으로 numeric 변환, Age는 그대로 numeric
users['Gender_binary'] = users['Gender'].map({'M':1,'F':0})
user_features = users[['UserID','Gender_binary','Age']]
# ratings와 user_features 머지
merged_data = pd.merge(ratings, movies[['MovieID']], on='MovieID', how='inner')
merged_data = pd.merge(merged_data, user_features, on='UserID', how='left')


# 라벨 생성: rating>=4 선호(1), 아니면 비선호(0)
merged_data['Label'] = (merged_data['Rating'] >= 4).astype(int)

feature_matrix = []
labels = []

for _, row in merged_data.iterrows():
    # 영화 벡터: 최종 영화벡터(영화 + 서브장르 반영)
    movie_vec = movie_final_vector_cache.get(row['MovieID'], np.zeros(embedding_dim))
    
    # 사용자 취향 벡터
    user_vec = user_preference_cache.get(row['UserID'], np.zeros(embedding_dim))
    
    # 사용자 정보(나이, 성별)
    user_gender = row['Gender_binary'] if not pd.isnull(row['Gender_binary']) else 0
    user_age = row['Age'] if not pd.isnull(row['Age']) else 0
    # 최종 벡터 = 영화벡터 + 사용자취향벡터 + 사용자정보
    combined_vec = np.concatenate([movie_vec, user_vec, np.array([user_gender, user_age])])
    
    feature_matrix.append(combined_vec)
    labels.append(row['Label'])

feature_matrix = np.array(feature_matrix)
labels = np.array(labels)

# 데이터셋 분할: Train(80%), Test(20%)
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, labels, test_size=0.2, random_state=42)


# Logistic Regression 모델 학습 및 평가
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga')
clf_lr.fit(X_train_scaled, y_train)
y_pred_lr = clf_lr.predict(X_test_scaled)


print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_lr))


# RandomForestClassifier 모델 생성 및 학습
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, 
                                max_depth=None, 
                                class_weight='balanced', 
                                random_state=42, 
                                n_jobs=-1)

clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)

# 평가 결과 출력
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))