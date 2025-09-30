(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        red - block
        purple - block
        yellow - block
        blue - block
        green - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable red)
        (clear red)
        (ontable purple)
        (on yellow purple)
        (on blue yellow)
        (on green blue)
        (on orange green)
        (clear orange)
        ; red
        ; purple -> yellow -> blue -> green -> orange
    )
    (:goal (and (on purple green)
        (on blue purple)
        (on orange blue)
        (on red orange)
        (on yellow red)))
)
    