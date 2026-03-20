CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT DEFAULT 'user',    -- admin, staff, guest
    enabled BOOLEAN DEFAULT true
);
CREATE TABLE face_embeddings (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(id),
    embedding REAL[] NOT NULL
);

ALTER TABLE persons
ADD CONSTRAINT unique_person_name UNIQUE (name);

ALTER TABLE face_embeddings
ADD CONSTRAINT unique_face_per_person UNIQUE (person_id, embedding);

ALTER TABLE face_embeddings
DROP CONSTRAINT unique_face_per_person;


select * from persons
select * from face_embeddings