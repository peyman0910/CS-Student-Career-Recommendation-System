from flask import Flask, jsonify, request
import pickle
import numpy as np
#import sklearn
import pandas as pd

import joblib



#model = pickle.load(open('svm_model.pkl','rb'))
loaded_model = joblib.load('career_recommendation_model.pkl')
#model = joblib.load('RandomForest_model.pkl')

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
    
    # Create a mock student object for explanation
    mock_student = {
        'Student ID': 'NEW',
        'Interested Domain': student_data['Interested Domain'],
        'Python': student_data['Python'],
        'SQL': student_data['SQL'],
        'Java': student_data['Java'],
        'Projects': student_data['Projects'],
        'UK Degree Classification': student_data['UK Degree Classification']
    }

    #explanation = create_explanation(mock_student, top_career, career_info, confidence)

    return top3_careers, top3_scores#, explanation


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    student={
        'Student ID': 'NEW',
        'Interested Domain':request.form.get ('domain'),
        'Python': request.form.get('python'),
        'SQL': request.form.get('sql'),
        'Java':request.form.get('java'),
        'Projects':request.form.get('project'),
        'UK Degree Classification': request.form.get('degree')
    }
    result = recommend_for_new_student(student)
    return jsonify({'top3_careers':result})
    '''
    result = model.predict(input_query)
    input_query = np.array([[humidity, temperature, step_count, respiratery_rate, heart_rate]])
    print(input_query)
    #input_query = [[humidity, temperature, step_count, respiratery_rate, heart_rate]]
    #df1 = pd.DataFrame(input_query)
    result = recommend_for_new_student(student)
    return jsonify({'careers':result})
  
    #result = {'humidity':humidity, 'temperature':temperature, 'step_count':step_count, 'stress_level':stress_level, 'respiratery_rate':respiratery_rate, 'heart_rate':heart_rate}
    print(result[0])
    #return jsonify(str(result))
    return jsonify({'stress_level':str(result)})

    '''

if __name__ == '__main__':
    app.run(debug=True)
    '''
    student_data = {
            'domain': "Cloud Computing",
            'python': "Weak",
            'sql': "Weak",
            'java': "Weak",
            'project': "3D Animation",
            'degree': "First-Class Honours"
        }

    mock_student = {
        'Student ID': 'NEW',
        'Interested Domain': student_data['domain'],
        'Python': student_data['python'],
        'SQL': student_data['sql'],
        'Java': student_data['java'],
        'Projects': student_data['project'],
        'UK Degree Classification': student_data['degree']
    }

    print(recommend_for_new_student(mock_student))
    '''

