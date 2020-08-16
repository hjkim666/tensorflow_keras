import random
rnd = random.randint(1,100) #1~100사이 난수

while True:
    usernum = input("1~100사이 숫자를 입력하세요: ")
    if rnd == int(usernum):
        print('추카추카.정답입니다')
        break
    elif rnd < int(usernum):
        print('낮춰주세요')
    else:
        print('높여주세요')
print('게임 종료')