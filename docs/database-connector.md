# The database connector

Every query in weaveio starts with a database connector - `Data`.
The `Data` class represents all the information in the database and contains the backend for connecting to it.

When designing the database schema, you inherit from the `Data` class. 
However, for convenience, when you import `Data` from weaveio, the opr3 database connector is imported, not the base `Data` class.

`Data` has the following signature:
```python
Data(rootdir: Union[Path, str] = None, host: str = None, port: int = None, 
     dbname: str = None, password: str = None, user: str = None, verbose=False)
```
Most of these arguments will have default values which have been specified (for example in the opr3 database connector class).
However, you can override any of them if necessary and `user` and `password` are required arguments.
You can also specify environment variables to avoid pushing passwords to github.
These are:
```python
WEAVEIO_DB
WEAVEIO_HOST
WEAVEIO_PORT
WEAVEIO_PASSWORD
WEAVEIO_USER
WEAVEIO_ROOTDIR
```

