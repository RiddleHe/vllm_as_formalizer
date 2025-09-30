(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        red - block
        purple - block
        blue - block
        yellow - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on red green)
        (on purple red)
        (clear purple)
        (ontable blue)
        (on yellow blue)
        (on orange yellow)
        (clear orange)
        ; green -> red -> purple
        ; blue -> yellow -> orange
    )
    (:goal (and (on orange yellow)
        (on red orange)
        (on green red)
        (on blue green)
        (on purple blue)))
)
    