
import sqlite3
from datetime import datetime
import time

timeframe = '2019-04'
sql_transaction = []
connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()


def transaction_bldr():
    global sql_transaction
    c.execute('BEGIN TRANSACTION')
    for s in sql_transaction:
        try:
            c.execute(s)
        except:
            pass
    connection.commit()
    sql_transaction = []
      

def sql_finding():
    sql = 'SELECT comment_id from parent_reply WHERE parent_id IS NULL LIMIT 10000'
    c.execute(sql)
    results = []
    for _ in range(10000):
        results.append(c.fetchone()[0])

    return results


def sql_pull_out_comment(parent):
    try:
        sql = 'SELECT comment, score from parent_reply WHERE comment_id = "{}" LIMIT 1'.format(parent)
        c.execute(sql)
        result = c.fetchone()
        if result[0] != None:
            return result
        else: return False
    except: return False 


def sql_pull_out_child(parent):
    try:
        sql = 'SELECT child from parent_reply WHERE comment_id = "{}" LIMIT 1'.format(parent)
        c.execute(sql)
        result = c.fetchone()
        if result[0] != None:
            return result[0]
        else: return False
    except: return False 


def sql_delete(comment_id):
    global sql_transaction
    try:
        sql = 'DELETE FROM parent_reply WHERE comment_id = "{}"'.format(comment_id)
        sql_transaction.append(sql)
    except Exception as e:
        print('s0 insertion',str(e))


if __name__ == '__main__':
    counter = 0
    ids = []
    comments = []

    f = open('place/' + timeframe, 'a', encoding='utf8')

    while True:
        try:

            parent_ids = sql_finding()
            print('lol')
            for parent in parent_ids:
                while parent:
                    ids.append(parent)
                    parent = sql_pull_out_child(parent)

                for comment_id in ids:
                    comment = sql_pull_out_comment(comment_id)
                    comments.append(comment)
                    sql_delete(comment_id)

                for comment in comments:
                    if comment:
                        f.write(comment[0] + " " + str(comment[-1]) + '\n')
                        counter += 1
                
                ids = []
                comments = []

                f.write('<end examples>' + '\n')

            transaction_bldr()

            if counter % 100 == 0:
                print('Comments all = {}, Time: {}'.format(counter, str(datetime.now())))

        except Exception as e:
            print(str(e))
            break