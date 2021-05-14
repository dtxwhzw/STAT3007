# STAT3007
### Introduction
This is the Project of UQ STAT3007, we used the BERT model to build a sentiment classifier based on some
app's reviews data.

The dataset is [Google Play Apps Reviews](https://www.kaggle.com/therealsampat/google-play-apps-reviews?select=apps.csv), which
contain 15 apps' reviews. They're 
1. Any.do: To do list, Reminders, Planner & Calendar
2. Todoist: To-Do List, Tasks & Reminders
3. TickTick: ToDo List Planner, Reminder & Calendar
4. Habitica: Gamify Your Tasks
5. Forest: Stay focused
6. Habit Tracker
7. Do It Now: RPG To Do List. Habit Tracker. Planner
8. HabitNow - Daily Routine, Habits and To-Do List
9. Microsoft To Do: List, Task & Reminder
10. Sectograph. Planner & Time manager on clock widget
11. TimeTune - Optimize Your Time, Productivity & Life
12. Artful Agenda - Plan, Sync & Organize in Style
13. Tasks: Todo list, Task List, Reminder
14. Business Calendar 2 - Agenda, Planner & Widgets
15. Planner Pro - Personal Organizer

The reviews of these apps have scores from 1 to 5. Because of the inblanace of data, i
convert these scores into 3 classes, which are negative(1,2), neural(3), positive(4,5).
In addition, I divided each category of data into train set, test set, valid set in an 8:1:1 ratio.
And they are in the [Data/offline_data/sentiment](Data/offline_data/sentiment)

### How to use
```
git clone https://github.com/dtxwhzw/STAT3007.git
pip install requirements.txt
# train model
python train.py conf/train_conf.json
# evaluate model
python train.py conf/train_conf.json
# inference
python inference.py conf/train_conf.json
# you can replace the input_str in line 84 with whatever sentence as you want
```

### Tips
The training data are in the [Data/offline_data/sentiment](Data/offline_data/sentiment).

And the bert pretrain file should in the [Data/online_data/saved_models/bert-base-cased](Data/online_data/saved_models/bert-base-cased)

The train procedure will need 14555MB RAM in GPU. So, if you encounter the CUDA out of memory issue,
you can reduce the batch size into 16 or 8 in [configuration file](conf/train_conf.json).

If you encounter some network issue while load the pretrain models.
You can download the bert file from "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
and the vocab file from "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt"

### Result
The f1 score of the [first configuration file](conf/train_conf.json) is 0.84
as for the [second configuration file](conf/train_conf_v1.json) is 0.86.

You can switch the parameters in the configuration file and train some new models. It is reocommend to switch the *max_seq_len* in [200,500],
 *drop_out* , *lr*, and *batch* parameters(the batch size larger the result would be better).