# inside urls.py
from django.urls import path
from django.http import HttpResponse
from .views import run_bl_view

urlpatterns = [
    # path('run_rl/', run_bl_view, name='run_rl'),
        # path('favicon.ico', lambda request: HttpResponse(status=204)),
]
