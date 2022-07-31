from django.contrib import admin
from django.contrib import admin
from datadb.models import user

# admin.site.index_template = '3.html'
admin.site.index_template = 'muban2.html'
admin.site.index_title = '视窗'
admin.site.site_header = '数据库管理'
admin.site.site_title = 'ccc'
admin.site.login_template = 'login.html'
admin.site.logout_template='logout.html'