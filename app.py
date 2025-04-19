from flask import Flask, jsonify, request
import pickle
import numpy as np
import pandas as pd


with open('career_recommendation_model.pkl', 'rb') as f:
        loaded_model  = pickle.load(f)

U = loaded_model['U']
sigma = loaded_model['sigma']
Vt = loaded_model['Vt']
predictions = loaded_model['predictions']
mappings = loaded_model['mappings']
career_info = loaded_model['career_info']
user_means = loaded_model['user_means']


app = Flask(__name__)


def recommend_for_new_student(student_data):
    """Generate recommendations for a new student"""
    skill_mapping = {'Weak': 1, 'Average': 2, 'Strong': 3}
    career_scores = {}

    for career, info in career_info.items():
        domain_match = 0.0
        if student_data['Interested Domain'] in info['domains']:
            domain_match = 1.0

        skill_match = 0.0
        skill_count = 0

        if info['python_mode'] == student_data['Python']:
            skill_match += 1.0
            skill_count += 1

        if info['sql_mode'] == student_data['SQL']:
            skill_match += 1.0
            skill_count += 1

        if info['java_mode'] == student_data['Java']:
            skill_match += 1.0
            skill_count += 1

        if skill_count > 0:
            skill_match = skill_match / skill_count

        # Project match (weight - 0.25)
        project_match = 0.0
        if student_data['Projects'] in info['projects']:
            project_match = 1.0

        # Degree match (weight - 0.15)
        degree_match = 0.0
        if info['degree_mode'] == student_data['UK Degree Classification']:
            degree_match = 1.0

        # Calculate weighted score
        career_scores[career] = (
            0.35 * domain_match +
            0.25 * skill_match +
            0.25 * project_match +
            0.15 * degree_match
        )

    # Get top 3 careers
    sorted_careers = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)
    top3_careers = [career for career, score in sorted_careers[:3]]
    top3_scores = [score for career, score in sorted_careers[:3]]

    # Create explanation for top career
    top_career = top3_careers[0]
    confidence = 0.90 + 0.09 * top3_scores[0]  # Scale to 90-99%

    return top3_careers, top3_scores#, explanation


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        student={
            'Student ID': 'NEW',
            'Interested Domain':request.form.get ('domain'),
            'Python': request.form.get('python'),
            'SQL': request.form.get('sql'),
            'Java':request.form.get('java'),
            'Projects':request.form.get('project'),
            'UK Degree Classification': request.form.get('degree')
        }
        #print("student data ",student)
        result = recommend_for_new_student(student)
        print(result)
        return jsonify({'top3_careers':result[0]})
   

if __name__ == '__main__':
    app.run(debug=True)
    