from django.contrib import admin
from django.urls import path 
from .views import *
urlpatterns = [
    path('',page_1),
    path('page2',page_2),
]
