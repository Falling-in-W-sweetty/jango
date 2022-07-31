from django.conf.urls import url
from . import views  # 其中点表示上级目录
# app_name = 'movie'
from django.urls import path
urlpatterns = [
    path('info/<table_id>/', views.userinfo,name='info'),  ###增加此路由信息
    path('addrow/<table_id>/', views.useradd,name='addrow'),  ###增加此路由信息
    path('editrow/<table_id>/', views.useredit,name='editrow'),  ###添加useredit路由
    path('delrow/<table_id>/', views.userdel,name='delrow'),  ####a添加userdel路由
]
