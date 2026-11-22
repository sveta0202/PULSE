# Эксперименты классификатора действия (leave-one-day-out CV)

| tag | accuracy | logloss | non-default args |
|---|---|---|---|
| smoke | 0.9625 | 0.293 | steps=800, crop=500, inmode=none, feat=env, seeds=1 |
| s3 | 0.9625 | 0.2949 | steps=800, crop=500, inmode=none, feat=env, seeds=3 |
| crop1000 | 0.9500 | 0.3463 | steps=800, crop=1000, inmode=none, feat=env, seeds=3 |
| center | 0.9500 | 0.3903 | steps=800, crop=500, inmode=center, feat=env, seeds=3 |
| envframe | 0.9375 | 0.3646 | steps=800, crop=500, inmode=none, feat=envframe, seeds=3 |
| s3_1500 | 0.9625 | 0.3212 | steps=1500, crop=500, inmode=none, feat=env, seeds=3 |
