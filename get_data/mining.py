#JSON TO DATABASE

import sqlite3
import json
from datetime import datetime
import time

timeframe = '2019-03'
sql_transaction = []
cleanup = 2500000

connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, child TEXT, comment TEXT, score INT)")


def format_data(data):
    data = data.replace('\n',' ').replace('\r',' ').replace('"',"'")
    return data


def transaction_bldr(sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 70000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


def acceptable(data):
    if len(data.split()) > 256 or len(data) < 1:
        return False
    elif 'http' in data:
        return False
    elif '&gt' in data:
        return False    
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


def find_parent(pid):
    try:
        sql = "SELECT comment_id FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result[0] != None:
            return True
        else: return False
    except: return False


def find_parents_child(pid):
    try:
        sql = "SELECT child FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql)
        result = c.fetchone()
        if result[0] != None:
            return result[0]
        else: return False
    except: return False


def child_score(childid):
    try:
        sql = "SELECT score FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(childid)
        c.execute(sql)
        result = c.fetchone()
        if result[0] != None:
            return result[0]
        else:
            return False
    except: return False        


def sql_insert_no_parent(commentid, comment, score):
    try:
        sql = """INSERT INTO parent_reply (comment_id, comment, score) VALUES ("{}","{}", {});""".format(commentid, comment, score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))


def update_parent(parentid, commentid):
    try:
        sql = "UPDATE parent_reply SET child = '{}' WHERE comment_id = '{}';".format(commentid, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))


def sql_insert_has_parent_without_child(parentid, commentid, comment, score):
    try:
        sql =  """INSERT INTO parent_reply (parent_id, comment_id, comment, score) VALUES ("{}","{}", "{}", {});""".format(parentid, commentid, comment, score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))


def sql_insert_has_parent(parentid, commentid, comment, score, childid):
    try:
        sql = """UPDATE parent_reply SET parent_id = '{}', comment_id = '{}', comment = '{}', score = {} WHERE parent_id ='{}';""".format(parentid, commentid, comment, score, childid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion',str(e))

if __name__ == '__main__':
    create_table()
    row_counter = 0
    childs = 0
    with open('RC_{}'.format(timeframe), buffering = 1000) as f:
        for row in f:
            row_counter += 1
            try:
                if row_counter >= 0:
                    row = json.loads(row)
                    comment_id = row['id']
                    parent_id = row['parent_id'][3:]
                    comment = format_data(row['body'])
                    score = row['score']
                    if acceptable(comment):
                        if find_parent(parent_id):
                            child_id = find_parents_child(parent_id)
                            if child_id:
                                ex_child_score = child_score(child_id)
                                if ex_child_score:
                                    if ex_child_score < score:
                                        update_parent(parent_id, comment_id)
                                        sql_insert_has_parent(parent_id, comment_id, comment, score, child_id)
                                else:
                                    update_parent(parent_id, comment_id)
                                    sql_insert_has_parent(parent_id, comment_id, comment, score, child_id)
                                    childs += 1              
                            else:                            
                                update_parent(parent_id, comment_id)
                                sql_insert_has_parent_without_child(parent_id, comment_id, comment, score)
                                childs += 1                
                        else:                         
                            sql_insert_no_parent(comment_id, comment, score)    
            except Exception as e:
                print(str(e))
            
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, childs: {}, Time: {}'.format(row_counter, childs, str(datetime.now())))
          
            if row_counter % cleanup == 0:
                print("Cleanin up!")
                sql = "DELETE FROM parent_reply WHERE child IS NULL AND parent_id IS NULL"
                c.execute(sql)
                connection.commit()
                c.execute("VACUUM")
                connection.commit()           

