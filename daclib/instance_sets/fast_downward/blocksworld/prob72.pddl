

(define (problem BW-rand-17)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 )
(:init
(arm-empty)
(on b1 b8)
(on b2 b7)
(on b3 b5)
(on-table b4)
(on b5 b14)
(on-table b6)
(on-table b7)
(on b8 b10)
(on b9 b2)
(on b10 b12)
(on b11 b3)
(on b12 b17)
(on b13 b1)
(on b14 b9)
(on-table b15)
(on b16 b15)
(on b17 b11)
(clear b4)
(clear b6)
(clear b13)
(clear b16)
)
(:goal
(and
(on b2 b14)
(on b3 b17)
(on b4 b5)
(on b5 b13)
(on b6 b8)
(on b7 b2)
(on b9 b6)
(on b10 b15)
(on b12 b4)
(on b13 b9)
(on b14 b10)
(on b15 b1)
(on b16 b3))
)
)

