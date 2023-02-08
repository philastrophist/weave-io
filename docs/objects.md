# Objects and their attributes in weaveio with cats

!!! info
    This section will talk about how to access objects and their attributes in weaveio.
    It will talk about schema and relationships but it will not describe how to build a schema or write data to the database.
    This is done in the [writing](writing.md) section.

An object in weaveio is a container for attributes.
For example, a `Cat` object has a `tail_length` attribute. 
A `Cat` object may also have an `Owner` attribute (note that the `Owner` is itself an object that has its own attributes).

Let's suppose that the database has this structure:

``` mermaid
graph LR
  Owner --> Cat;
  Cat --> Toy;  
```
This structure is pronounced *"An Owner has a Cat which has a Toy"*.

In weaveio you always traverse to objects from another object following these arrows.

1. For each cat get its owner: `cat.owner`
2. For each cat's owner get its name and return them all: `cat.owner.name()`

You can read the dots as *"For each cat get its owner, for each owner get its name"*.

You cannot return an object in weaveio, only attributes of objects.

!!! success
    ```python
    >>> database.cats.owner['name']()
    ['Andrea', 'Mike', 'Humphrey']
    ```

!!! failure
    
    ```python
    >>> database.cats.owner()
    SyntaxError: Owner cannot be returned/identified since it doesn't define any unique idname. 
             If you want to return all singular data for Owner use ...['*']
    ```

### Attribute access
In weaveio you can access attributes using either the square brackets `object['attribute']` or the dot syntax `object.attribute`. 
There is no difference between them. 
The only reason that you would use `object['attribute']` over `object.attribute` is when requesting more than one attribute at once (see [tables](tables.md)).

Weaveio is designed such that, in the relationship hierarchy above, *all linked objects/attributes are accessible*.

So `owner.cat.toy` is the same as `owner.toy`:
!!! success
    ``` mermaid
    graph LR
      Owner --> Cat;
      Cat --> Toy;  
    ```
    ```python
    >>> owner.cat.toy() == owner.toy()
    True
    ```

Furthermore, `owner.squeakiness` is the same as `owner.toy.squeakiness`:
!!! success
    ``` mermaid
    graph LR
      Owner --> Cat;
      Cat --> Toy;  
    ```
    ```python
    >>> owner.toy.squeakiness() == owner.squeakiness()
    True
    ```

However, `cat.bike` is not valid because the cat has no direct access to the bike. 
You must explicitly traverse to Owner first:
!!! warning
    ``` mermaid
    graph LR
      Owner --> Cat;
      Owner --> Bike;
      Cat --> Toy;  
    ```
    ```python
    >>> cat.bike
    AmbiguousPathError
    >>> cat.owner.bike    
    ```



### A note on pluralisations
If an object has more than one linked object then you must use the plural form of the object. 

Let's consider an altered schema where there are multiple relationships between each object type.
``` mermaid
graph LR
  Owner --"many"--> Cat;
  Cat --"one"--> Owner;
  Cat --"many"--> Toy;
  Toy--"one"-->Cat;
```

!!! success
    ```python
    >>> cat.toys.name()
    ['mouse', 'ribbon', 'bell']
    ```

!!! failure
    ```python
    >>> cat.toy.name()
    KeyError: A Cat has more than one Toy. Use the plural "toys".
    ```

!!! success
    ```python
    >>> toy.cat.name()
    Felix
    ```

## Cardinalty
If we interpret the dot syntax as "for each" as detailed above, then this implies an increase in the number of results with each level.
For example, in the above example each cat has one owner and an owner can only own one cat. 
Let's now say that Mike owns both Felix and Whiskers.

The list of owners is still the same:
```python
>>> database.owners['name']()
['Andrea', 'Mike', 'Humphrey']
```

But now if we ask for the owner of each cat:
```python
>>> database.cats.owner['name']()
['Andrea', 'Mike', 'Humphrey', 'Mike']
```
We can now see that Mike appears twice.
    
!!! warning
    Things can quickly get out of control in a large interconnected database when traversing the same relationship repeatedly: 
    ```python
    >>> database.cats.owner.cats.owner['name']()
    ['Andrea', 'Mike', 'Mike', 'Humphrey', 'Mike', 'Mike']
    ```
    Things duplicate because we've asked for the owner of each cat and then the owner of each of those cats.


## Types of object
Objects have types and you can query different types just as you would normally. 
In weaveio all objects inherit from `Hierarchy`:

```python
>>> database.hierarchies['name']()
['Andrea', 'Mike', 'Humphrey', 'Felix', 'Whiskers', 'Smudge', 'mouse', 'ribbon', 'bell']
```
If `Cat` is defined as a type of `Animal` then `database.animals['name']()` will return all the names of all the Cats:

```python
>>> database.animals['name']()
['Smudge', 'Felix', 'Whiskers']
```
If Owner was defined as an animal then this query would also fetch the names of the owners too.


