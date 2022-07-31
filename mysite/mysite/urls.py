"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
# from gongxiantu import views as views1
# from gongxiantu2 import views as views2
# from tuihuo import views as views3

urlpatterns = [
    path('admin/', admin.site.urls),
    # path('gongxiantu/', views1.index),
    # path('gongxiantu2/', views2.index),
    # path('tuihuo/', views3.index),
    path('database/', include('datadb.urls'))
]
