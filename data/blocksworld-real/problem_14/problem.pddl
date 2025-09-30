(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        red - block
        orange - block
        blue - block
        yellow - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on red green)
        (on orange red)
        (clear orange)
        (ontable blue)
        (on yellow blue)
        (on purple yellow)
        (clear purple)
        ; green -> red -> orange
        ; blue -> yellow -> purple
    )
    (:goal (and (on yellow orange)
        (on purple yellow)
        (on red green)
        (on blue red)))
)
    