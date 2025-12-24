CREATE TABLE person (
    roll_number  VARCHAR(20) PRIMARY KEY,
    name         VARCHAR(100)
);

INSERT INTO person (roll_number, name) VALUES
('UU1', 'aashu'),
('UU3', 'aman'),
('UU19', 'swapnil');

CREATE TABLE face_embedding (
    embedding_id SERIAL PRIMARY KEY,
    roll_number  VARCHAR(20) NOT NULL,
    embedding    FLOAT4[] NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (roll_number)
        REFERENCES person(roll_number)
        ON DELETE CASCADE
);

CREATE TABLE attendance (
    attendance_id SERIAL PRIMARY KEY,
    roll_number     VARCHAR(20) NOT NULL,
    attendance_date DATE NOT NULL DEFAULT CURRENT_DATE,
    marked_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (roll_number)
        REFERENCES person(roll_number),

    UNIQUE (roll_number, attendance_date)
);
CREATE INDEX idx_face_embedding_roll_number
    ON face_embedding (roll_number);
