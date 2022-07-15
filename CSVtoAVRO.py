import pandas as pd
from fastavro import writer, reader, parse_schema

df = pd.read_csv('/Users/sudarshangupta/Downloads/Book5.csv')
df.head()


schema= {
"type": "record",
"name": "Cpu",
"namespace" : "my.example",
"fields": [
     {"name": "name", "type": "string"},
     {"name": "time", "type": "int"},
     {"name": "cpu", "type": "string"},
     {"name": "dc", "type": "string"},
     {"name": "deployment", "type": "string"},
     {"name": "env", "type": "string"},
     {"name": "host", "type": "string"},
     {"name": "region", "type": "string"},
     {"name": "role", "type": "string"},
     {"name": "usage_guest", "type": "double"},
     {"name": "usage_guest_nice", "type": "double"},
     {"name": "usage_idle", "type": "double"},
     {"name": "usage_iowait", "type": "double"},
     {"name": "usage_irq", "type": "double"},
     {"name": "usage_nice", "type": "double"},
     {"name": "usage_softirq", "type": "double"},
     {"name": "usage_steal", "type": "double"},
     {"name": "usage_system", "type": "double"},
     {"name": "usage_user", "type": "double"},
     {"name": "zone", "type": "string"}
    ]
}

parsed_schema = parse_schema(schema)

records = df.to_dict('records')

with open('june27.avro', 'wb') as out:
    writer(out, parsed_schema, records)
