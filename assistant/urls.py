from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask/', views.ask_question, name='ask_question'),
    path('status/', views.system_status, name='system_status'),
]
