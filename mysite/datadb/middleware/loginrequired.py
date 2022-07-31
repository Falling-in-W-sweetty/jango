from django.shortcuts import redirect
from django.conf import settings


class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.login_url = settings.LOGIN_URL
        self.open_urls = [self.login_url] + getattr(settings, 'OPEN_URLS', [])

    def __call__(self, request):
        print(request.user)
        if not request.user.is_authenticated and request.path_info not in self.open_urls:
            # 此处调用request对象的user的is_authenticated方法不同于其他验证的request.GET.get(user)
            return redirect(self.login_url + '?next=/admin/')
            # return redirect(self.login_url + '?next=/3/' + request.path)
        return self.get_response(request)
