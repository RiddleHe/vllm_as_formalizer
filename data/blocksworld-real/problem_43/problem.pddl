(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        yellow - block
        red - block
        purple - block
        orange - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (clear green)
        (ontable yellow)
        (clear yellow)
        (ontable red)
        (clear red)
        (ontable purple)
        (on orange purple)
        (clear orange)
        (ontable blue)
        (clear blue)
        ; green
        ; yellow
        ; red
        ; purple -> orange
        ; blue
    )
    (:goal (and (on blue yellow)
        (on orange blue)
        (on green orange)
        (on purple red)))
)
    