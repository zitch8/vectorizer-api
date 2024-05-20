from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = FastAPI()

class TagsInput(BaseModel):
    tags: list[int]

@app.post("/recommend")
def post_recommendations(tags: TagsInput):
    df = pd.read_csv('Food_Dataset.csv')
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vector = cv.fit_transform(df['tags']).toarray()
    
    store_food_tags = []
    for food in tags.tags:
        food_index = df[df['id_'] == food].index[0]
        food_tag = df.loc[food_index, 'tags']
        store_food_tags.append(food_tag)

    store_food_tags_vector = cv.transform([', '.join(store_food_tags)])
    cosine_similarities = cosine_similarity(vector, store_food_tags_vector).flatten()
    recommeded_indices = cosine_similarities.argsort()[-26:][::-1]
    recommended_recipes = df.iloc[recommeded_indices]

    response_data = []
    for _, recipe in recommended_recipes.iterrows():
        try:
            recipe_data = {
                "image_url": recipe['thumbnail_url'],
                "recipe_name": recipe['name'],
                "ingredients": recipe['cleaned_ingredients'],
                "protein": recipe['protein'],
                "fat": recipe['fat'],
                "calories": recipe['calories'],
                "sugar": recipe['sugar'],
                "carbohydrates": recipe['carbohydrates'],
                "fiber": recipe['fiber'],
                "instructions": recipe['cleaned_instructions'],
                "tags": recipe['tags']
            }

            response_data.append(recipe_data)

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            continue

    return response_data



class IngredientsInput(BaseModel):
    ingredients: str

@app.post("/search")
def post_ingredients(ingredients_input: IngredientsInput):
    new_df = pd.read_csv('Food_Dataset.csv')
    new_df['cleaned_ingredients'] = new_df['cleaned_ingredients'].apply(
        lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', ')
        )
    
    vectorizer = TfidfVectorizer()
    tfid_vectorizer = vectorizer.fit_transform(new_df['cleaned_ingredients'].apply(lambda x: ', '.join(x)))

    user_input = ingredients_input.ingredients.lower().split(', ')
    user_input_vector = vectorizer.transform([', '.join(user_input)])
    
    cosine_similarities = cosine_similarity(tfid_vectorizer, user_input_vector).flatten()
    
    indices = cosine_similarities.argsort()[-26:][::-1]

    recommended_recipes = new_df.iloc[indices]

    response_data = []
    for _, recipe in recommended_recipes.iterrows():
        try:
            recipe_data = {
                "image_url": recipe['thumbnail_url'],
                "recipe_name": recipe['name'],
                "ingredients": recipe['cleaned_ingredients'],
                "protein": recipe['protein'],
                "fat": recipe['fat'],
                "calories": recipe['calories'],
                "sugar": recipe['sugar'],
                "carbohydrates": recipe['carbohydrates'],
                "fiber": recipe['fiber'],
                "instructions": recipe['cleaned_instructions'],
                "tags": recipe['tags']
            }

            response_data.append(recipe_data)

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            continue
    
    return response_data

# For Main Recommendation


@app.get('/')
async def test():
    return {
        "hello": "world"
    }


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# @app.get('/')
# async def test():
#     return {
#         "hello": "world"
#     }