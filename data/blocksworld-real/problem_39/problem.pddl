(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        yellow - block
        red - block
        green - block
        blue - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (on yellow purple)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable green)
        (clear green)
        (ontable blue)
        (on orange blue)
        (clear orange)
        ; purple -> yellow
        ; red
        ; green
        ; blue -> orange
    )
    (:goal (and (on blue yellow)
        (on orange blue)
        (on red green)
        (on purple red)))
)
    