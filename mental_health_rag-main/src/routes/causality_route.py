# from flask import Blueprint, request, jsonify
# from src.controllers.index import causality_controllers

# causality_routes = Blueprint('causality_routes', __name__)

# @causality_routes.route('causality_routes',methods=['POST'])
# def chatbot():
#     try:
#         data = request.get_json()
#         resp,code = causality_controllers.causality(data)
#         return jsonify(resp),code
#     except Exception as e:
#         return jsonify({"message"f"An error occurred {e}"}),500