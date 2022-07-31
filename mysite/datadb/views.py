from django.core.paginator import Paginator, PageNotAnInteger, InvalidPage, EmptyPage
from django.db import transaction
from django.shortcuts import render
from django.shortcuts import render, HttpResponse, redirect, reverse
import xlrd

from datadb import models
from pathlib import Path
import os
from django.contrib.auth.decorators import login_required

BASE_DIR = Path(__file__).resolve().parent.parent  # 当前文件路径的父级的父级也就是C:\Users\su\PycharmProjects\untitled1

def useradd(request, table_id):
    fields = [field.name for field in eval('models.'+table_id)._meta.get_fields()]
    if request.method == "GET":
        return render(request, "datadb_templates/useradd.html", {'fields':fields})
    new_row=eval('models.'+table_id)()
    for i in fields:
        new_row.__dict__[i]=request.POST.get(i)
    new_row.save()
    # user.objects.create(value1=value1, value2=value2, value3=value3)  ##新建用户
    return HttpResponse("/userinfo/")  ##添加成功后，重定向到userinfo界面


def useredit(request, table_id):
    if not request.user.is_superuser:
        return HttpResponse("你没有权限进行该操作")
    uid = request.GET.get("uid")  ##根据用户id进行查找更新
    user_data = eval('models.'+table_id).objects.values().filter(id=uid)#将table中的筛选对象用value转换成字典组成的列表
    if request.method == "GET":
        return render(request, "datadb_templates/useredit.html", {"user_data": user_data})
    edit_object=eval('models.'+table_id).objects.get(id=uid)#edit_objects代表要操作对象的，属于model类
    for i in user_data[0].keys():
        edit_object.__dict__[i]=request.POST.get(i)
    edit_object.save()


def userdel(request, table_id):
    if not request.user.is_superuser:
        return HttpResponse("你没有权限进行该操作")
    # user_list =exec(table_id+'.objects.values()')
    uid = request.GET.get("uid")  ###根据uid删除用户
    eval('models.'+table_id).objects.filter(id=uid).delete()
    return redirect(reverse('info', kwargs={'table_id': table_id}))  ###已经无用了；删除之后重定向到/userinfo/界面


def userinfo(request, table_id):
    if request.method == "POST":  # 请求方法为POST时，进行处理
        print('post')
        myFile = request.FILES.get("myfile", None)  # 获取上传的文件，如果没有文件，则默认为None 这里myfile对应前端元素名称
        if not myFile:
            return HttpResponse("未选择文件")
        excel_type = myFile.name.split('.')[1]
        if excel_type in ['xlsx', 'xls']:
            destination = open(os.path.join(BASE_DIR, 'static', myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
            for chunk in myFile.chunks():  # 分块写入文件
                destination.write(chunk)
            destination.close()
            # return HttpResponse("上传成功!")
            wb = xlrd.open_workbook(filename=os.path.join(BASE_DIR, 'static', myFile.name), file_contents=myFile.read())
            table = wb.sheets()[0]
            rows = table.nrows  # 总行数
            fields = [field.name for field in eval('models.'+table_id)._meta.get_fields()]  #全部字段值
            try:
                with transaction.atomic():  # 控制数据库事务交易
                    for i in range(1, rows):
                        rowVlaues = table.row_values(i)
                        # major = models.TMajor.objects.filter(majorid=rowVlaues[1]).first()
                        new_row=eval('models.'+table_id)()#新建一行
                        for index,i in enumerate(fields):
                            new_row.__dict__[i] = rowVlaues[index]
                        new_row.save()
                        # user.objects.create(value1=rowVlaues[0], value2=rowVlaues[1], value3=rowVlaues[2])
            except:
                return HttpResponse('解析excel文件或者数据插入错误')
            return HttpResponse('提交成功')
        else:  # else与最近的同级if配对 就近原则
            return HttpResponse('上传文件类型错误')
    else:
        table_all=[]
        for keys, value in models.__dict__.items():
            if type(value).__name__ == 'ModelBase':
                table_all.append(keys)
        # table_all=['user','Houdu']# 这里是表的集合，其他部分已经写成通用模板，如果需要添加新表，导入model后在这里添加它的名称
        user_list = eval('models.'+table_id).objects.values()
        paginator = Paginator(user_list, 20)
        page_num = request.GET.get('p')
        try:
            Page = paginator.page(page_num)
        # todo: 注意捕获异常
        except PageNotAnInteger:
            # 如果请求的页数不是整数, 返回第一页。
            Page = paginator.page(1)
        except InvalidPage:
            # 如果请求的页数不存在, 重定向页面
            return HttpResponse('找不到页面的内容')
        except EmptyPage:
            # 如果请求的页数不在合法的页数范围内，返回结果的最后一页。
            Page = paginator.page(paginator.num_pages)
        return render(request, "datadb_templates/userinfo.html", {'Page': Page, 'paginator': paginator, 'table_id': table_id,
                                                       'table_all':table_all})
