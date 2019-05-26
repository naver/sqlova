#!/usr/bin/env python

# Add a CSV file as a table into <split>.db and <split>.tables.jsonl
# Call as:
#   python add_csv.py <split> <filename.csv>
# For a CSV file called data.csv, the table will be called table_data in the .db
# file, and will be assigned the id 'data'.
# All columns are treated as text - no attempt is made to sniff the type of value
# stored in the column.

import argparse, csv, json, os
from sqlalchemy import Column, create_engine, MetaData, String, Table

def get_table_name(table_id):
    return 'table_{}'.format(table_id)

def csv_to_sqlite(table_id, csv_file_name, sqlite_file_name):
    engine = create_engine('sqlite:///{}'.format(sqlite_file_name))
    with open(csv_file_name) as f:
        metadata = MetaData(bind=engine)
        cf = csv.DictReader(f, delimiter=',')
        simple_name = dict([(name, 'col%d' % i) for i, name in enumerate(cf.fieldnames)])
        table = Table(get_table_name(table_id), metadata,
                      *(Column(simple_name[name], String())
                        for name in cf.fieldnames))
        table.drop(checkfirst=True)
        table.create()
        for row in cf:
            row = dict((simple_name[name], val) for name, val in row.items())
            table.insert().values(**row).execute()
    return engine

def csv_to_json(table_id, csv_file_name, json_file_name):
    with open(csv_file_name) as f:
        cf = csv.DictReader(f, delimiter=',')
        record = {}
        record['header'] = [(name or 'col{}'.format(i)) for i, name in enumerate(cf.fieldnames)]
        record['page_title'] = None
        record['types'] = ['text'] * len(cf.fieldnames)
        record['id'] = table_id
        record['caption'] = None
        record['rows'] = [list(row.values()) for row in cf]
        record['name'] = get_table_name(table_id)
        with open(json_file_name, 'a+') as fout:
            json.dump(record, fout)
            fout.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('file', metavar='file.csv')
    args = parser.parse_args()
    table_id = os.path.splitext(os.path.basename(args.file))[0]
    csv_to_sqlite(table_id, args.file, '{}.db'.format(args.split))
    csv_to_json(table_id, args.file, '{}.tables.jsonl'.format(args.split))
    print("Added table with id '{id}' (name '{name}') to {split}.db and {split}.tables.jsonl".format(
        id=table_id, name=get_table_name(table_id), split=args.split))

