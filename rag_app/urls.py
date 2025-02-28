from django.urls import path
from .views import upload_pdf, ask_question,index

urlpatterns = [
    path('', index, name='index'),
    path('upload_pdf/', upload_pdf, name='upload_pdf'),
    path('ask_question/', ask_question, name='ask_question'),
]