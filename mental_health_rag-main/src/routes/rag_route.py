from flask import Blueprint, request, jsonify
from src.controllers.index import chatbot_controllers

rag_routes = Blueprint('rag_routes', __name__)

@rag_routes.route('/chat_route',methods=['POST'])
def chatbot():
    try:
        data = request.get_json()
        resp,code = chatbot_controllers.generate_chat(data)
        print(resp)
        return jsonify(resp),code
    except Exception as e:
        return jsonify({"message"f"An error occurred {e}"}),500

@rag_routes.route('/response_route',methods=['POST'])
def chatbot1():
    try:
        data = request.get_json()
        resp,code = chatbot_controllers.generate_response(data)
        return jsonify(resp),code
    except Exception as e:
        return jsonify({"message"f"An error occurred {e}"}),500
