from django.urls import path

from django.contrib import admin
from django.urls import path
from home.views import *

urlpatterns = [
    path('', main, name='main'),
]
