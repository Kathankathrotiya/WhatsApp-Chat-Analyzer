import datetime
import re
import pandas as pd
from datetime import datetime

def preprocess(data):
    # Regular expression pattern to extract date, time, and message
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM]) - (.+)'

    # Find all occurrences of the pattern in the data
    matches = re.findall(pattern, data)

    message=[match[1] for match in matches]
    # Convert dates to 24-hour format
    dates_24hr = []
    for date_str in [match[0] for match in matches]:
        date_obj = datetime.strptime(date_str, "%d/%m/%y, %I:%M %p")
        date_24hr = date_obj.strftime("%d/%m/%y, %H:%M")
        dates_24hr.append(date_24hr)

    # print(dates_24hr)
    df = pd.DataFrame({'user_message': message, 'message_date': dates_24hr})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df
