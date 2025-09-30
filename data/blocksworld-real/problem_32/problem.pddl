(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        blue - block
        purple - block
        green - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (on blue yellow)
        (on purple blue)
        (clear purple)
        (ontable green)
        (on red green)
        (on orange red)
        (clear orange)
        ; yellow -> blue -> purple
        ; green -> red -> orange
    )
    (:goal (and (on green orange)
        (on red green)
        (on purple red)
        (on blue purple)
        (on yellow blue)))
)
    