from django.urls import path
from .views import upload_pdf, ask_question,index,knowledge_select,set_knowledge

urlpatterns = [
    path('', knowledge_select, name='select_knowledge'),  # Now this is the default page
    path('set_knowledge/', set_knowledge, name="set_knowledge"),
    path('index/', index, name='index'),  # Moved index here
    path('upload_pdf/', upload_pdf, name='upload_pdf'),
    path('ask_question/', ask_question, name='ask_question'),
    
]