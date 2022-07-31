import inspect

from mysite.wsgi import *
from django.contrib.auth.models import User
from datadb import models
user=User.objects.values()
# print(user.__dict__)
for keys,value in models.__dict__.items():
    if type(value).__name__ == 'ModelBase':
        print(keys)
print(len(models.__dict__.keys()))
print(len(inspect.getmembers(models)))
