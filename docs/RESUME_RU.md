# Как описать проект в резюме

Ссылка: [github.com/matveyvarfolomeeev/russian-command-routing](https://github.com/matveyvarfolomeeev/russian-command-routing).

## Короткий вариант

**Russian Command Routing — дообучение и оценка моделей выбора объектов по русским командам.** Сравнил bi-encoder retrieval, cross-encoder reranking и decision models; дообучил E5, Laya и Kev с LoRA. На новом каталоге из 177 объектов Kev получил 99,3% top-1 на 150 текстовых командах и 97,3% на их ASR-версиях. Опубликовал датасеты, код обучения, архитектуру и проверяемые результаты.

## Три пункта для ML-резюме

- Сформировал 22 656 синтетических обучающих примеров и наборы оценки русских команд; проверял пересечения запросов, объектов и групп, отдельно считал чистый текст и реальные ASR-расшифровки TTS.
- Реализовал сравнение E5 retrieval, RuModernBERT reranking и Laya/Kev/Nano decision models; провёл full fine-tuning и LoRA, выбор checkpoint по validation NLL и оценку на новом каталоге. Дообученный Kev: 149/150 clean и 146/150 ASR; исходный Kev: 143/150 в обоих случаях.
- Исследовал ошибки retrieval/selection, квантование и калибровку уверенности. Температура Laya снизила conditional NLL 0,404 → 0,241 и ECE 4,59% → 3,11% на известных отложенных группах, сохранив top-1. Опубликовал 4 400 прогнозов, журналы восьми запусков и скрипт пересчёта метрик.

**Стек:** Python, PyTorch, Transformers, PEFT/LoRA, ONNX Runtime, Kaggle GPU, pytest; Qt/QML и SQLite — для демонстрационного приложения.

## English

**Russian Command Routing — fine-tuning and evaluation for Russian object selection.** Compared bi-encoder retrieval, cross-encoder reranking and bounded decision models; fine-tuned E5/Laya and LoRA-adapted Kev. On a previously unseen 177-object catalog, Kev achieved 149/150 clean and 146/150 ASR top-1, versus 143/150 for the original checkpoint in both conditions. Published synthetic datasets, training code, architecture, run ledgers and 4,400 verifiable predictions.

## Что пояснить на собеседовании

ASR здесь — настоящие расшифровки синтезированной речи, не записи человеческих дикторов. Clean и ASR — связанные версии 150 команд. Разметка авторская, у каждого рецепта один seed. Новая оценка не использовалась для выбора исходных Laya/Kev checkpoint; последующие E5/reranker эксперименты уже смотрят известный набор. Отобранные веса видели только часть общего обучающего пула. Калибровка не повысила accuracy, а Brier ухудшился. Kev оказался сильнее по качеству на этом наборе, но его короткий CPU-прогон занял около 4,5 секунды на запрос.
