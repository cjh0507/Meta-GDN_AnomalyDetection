# GDN의 Abnormality Valuator의 구조를 설정하는 부분으로 보임. 현재 models.py에 정의된 SGC-based GDN과 완벽히 일치하는 세팅
def modelArch(in_feature, out_feature):

    config = [
        ('linear', [512, in_feature]),
        ('bn', [512]), # ! 추가된 것
        ('relu', [False]), # ! 추가된 것
        ('linear', [out_feature, 512])
    ]

    return config